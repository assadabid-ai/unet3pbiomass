import os
import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio as rio
from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings
from torchsummary import summary
from torch.utils.data import DataLoader
import json
# import wandb
from utils3p import calcuate_mean_std, stratify_data, freeze_encoder, BioMasstersDatasetS2S1, SentinelModel

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
torch.set_printoptions(sci_mode=False)

root_dir = os.getcwd() # Change to the folder where you stored preprocessed training data

S1_CHANNELS = {'2S': 8, '2SI': 12, '3S': 12, '4S': 16, '4SI': 24, '6S': 24}
S2_CHANNELS = {'2S': 20, '2SI': 38, '3S': 30, '4S': 40, '4SI': 48, '6S': 60}

df = pd.read_csv(os.path.join(f'./data/train_val_split_96_0.csv'), dtype={"id": str})

X_train, X_val, X_test = (df["id"].loc[df["dataset"] == 0].tolist(),
                          df["id"].loc[df["dataset"] == 1].tolist(),
                          df["id"].loc[df["dataset"] == 2].tolist())
print(df["dataset"].value_counts())
print("Total Images: ", len(df))

f = open('./data/mean.json')
mean = json.load(f)
f = open('./data/std.json')
std = json.load(f)
f = open('./data/mean_agb.json')
mean_agb = json.load(f)
f = open('./data/std_agb.json')
std_agb = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Empty cache
torch.cuda.empty_cache()
print(torch.version.cuda)
torch.backends.cudnn.benchmark = False

def ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply 3*3 convolutions with BN and ReLU.
    """
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels=in_channels if i == 1 else out_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 'same',
                         bias=not is_bn)  # Disable bias when using BatchNorm
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

class UNet3Plus(nn.Module):
    def __init__(self, input_shape, output_channels, deep_supervision=False, cgm=False, training=False):
        super(UNet3Plus, self).__init__()
        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and cgm
        self.training = training

        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = len(self.filters)
        self.upsample_channels = self.cat_blocks * self.cat_channels

        # Encoder
        self.e1 = ConvBlock(input_shape[0], self.filters[0])
        self.e2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[0], self.filters[1])
        )
        self.e3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[1], self.filters[2])
        )
        self.e4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[2], self.filters[3])
        )
        self.e5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[3], self.filters[4])
        )

        # Classification Guided Module
        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.filters[4], 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        ) if self.CGM else None

        # Decoder
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.filters[3], self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.final = nn.Conv2d(self.upsample_channels, output_channels, kernel_size=1)

        # Deep Supervision
        self.deep_sup = nn.ModuleList([
                ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False)
                for _ in range(3)
            ] + [ConvBlock(self.filters[4], output_channels, n=1, is_bn=False, is_relu=False)]
        ) if self.deep_supervision else None

    def forward(self, x) -> torch.Tensor:
        training = self.training
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # Classification Guided Module
        if self.CGM:
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # Decoder
        d4 = [
            F.max_pool2d(e1, 8),
            F.max_pool2d(e2, 4),
            F.max_pool2d(e3, 2),
            e4,
            F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = torch.cat(d4, dim=1)
        d4 = self.d4_conv(d4)

        d3 = [
            F.max_pool2d(e1, 4),
            F.max_pool2d(e2, 2),
            e3,
            F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=4, mode='bilinear', align_corners=True)
        ]
        d3 = [conv(d) for conv, d in zip(self.d3, d3)]
        d3 = torch.cat(d3, dim=1)
        d3 = self.d3_conv(d3)

        d2 = [
            F.max_pool2d(e1, 2),
            e2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=8, mode='bilinear', align_corners=True)
        ]
        d2 = [conv(d) for conv, d in zip(self.d2, d2)]
        d2 = torch.cat(d2, dim=1)
        d2 = self.d2_conv(d2)

        d1 = [
            e1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=16, mode='bilinear', align_corners=True)
        ]
        d1 = [conv(d) for conv, d in zip(self.d1, d1)]
        d1 = torch.cat(d1, dim=1)
        d1 = self.d1_conv(d1)
        d1 = self.final(d1)

        outputs = [d1]

        # Deep Supervision
        if self.deep_supervision and training:
            outputs.extend([
                F.interpolate(self.deep_sup[0](d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[1](d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[2](d4), scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[3](e5), scale_factor=16, mode='bilinear', align_corners=True)
            ])

        # Classification Guided Module
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]
        
        outputs = [F.ReLU(out) for out in outputs]
        
        if self.deep_supervision and training:
            return torch.cat(outputs, dim=0)
        else:
            return outputs[0]


ef train_base_model(suffix, encoder_name, encoder_weights, decoder_attention_type):
    # wandb.finish()    
    
    train_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                       s1_path=f"{root_dir}/train_features_s1_6S",
                                       agb_path=f"{root_dir}/train_agbm", X=X_train, mean=mean['6S'], std=std['6S'], 
                                       mean_agb=mean_agb, std_agb=std_agb, 
                                       transform=A.Compose([A.HorizontalFlip(), A.VerticalFlip(), 
                                                            A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]))

    val_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                     s1_path=f"{root_dir}/train_features_s1_6S",
                                     agb_path=f"{root_dir}/train_agbm", X=X_val, mean=mean['6S'], std=std['6S'], 
                                     mean_agb=mean_agb, std_agb=std_agb, transform=None)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_set, shuffle=False, batch_size=16, num_workers=8, pin_memory=True)

    in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S']
    INPUT_SHAPE = [in_channels, 320, 320]
    OUTPUT_CHANNELS = 1

    model = UNet3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, deep_supervision=True, cgm=False, training=True)
    '''model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, 
                             decoder_attention_type=decoder_attention_type,
                             in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S'], classes=1, activation=None)'''

    s2s1_model = SentinelModel(model, mean_agb=mean_agb, std_agb=std_agb, lr=0.001, wd=0.0001)

    # summary(s2s1_model.cuda(), (S2_CHANNELS['6S']+S1_CHANNELS['6S'], 256, 256)) 

    # wandb_logger = WandbLogger(save_dir=f'./models', name=f'{encoder_name}_6S_{decoder_attention_type}', 
    #                            project=f'{encoder_name}_6S_{decoder_attention_type}')

    ## Define a trainer and start training:
    on_best_valid_loss = ModelCheckpoint(filename="{epoch}-{valid/loss}", mode='min', save_last=True,
                                         monitor='valid/loss', save_top_k=2)
    on_best_valid_rmse = ModelCheckpoint(filename="{epoch}-{valid/rmse}", mode='min', save_last=True,
                                         monitor='valid/rmse', save_top_k=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = [on_best_valid_loss, on_best_valid_rmse, lr_monitor]

    # Initialize a trainer
    trainer = Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=200, 
                      # logger=[wandb_logger], 
                      callbacks=checkpoint_callback)
    # Train the model ⚡
    trainer.fit(s2s1_model, train_dataloaders=train_loader, val_dataloaders=val_loader)