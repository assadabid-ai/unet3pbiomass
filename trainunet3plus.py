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
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings
from torchsummary import summary
from torch.utils.data import DataLoader
import json
import wandb
from utils import calcuate_mean_std, stratify_data, freeze_encoder, BioMasstersDatasetS2S1, SentinelModel
import timm

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
torch.set_printoptions(sci_mode=False)

root_dir = "/home/aymen.tasneem/assad/biomassters_data" # Change to the folder where you stored preprocessed training data

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

def ConvBlock(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), 
              padding='same', is_bn=True, is_relu=True, n=2):
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels if i == 1 else out_channels, 
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 1,
                         bias=not is_bn)
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

class UNet3PlusTIMM(nn.Module):
    def __init__(self, 
                 backbone_name: str = "resnet50", 
                 output_channels: int = 1,
                 deep_supervision=False, 
                 cgm=False, 
                 training=False,
                 pretrained=True):
        super(UNet3PlusTIMM, self).__init__()

        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and cgm
        self.training = training

        # 1) Create the timm backbone, specifying features_only=True:
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(0, 1, 2, 3, 4)  # 5 stages
        )
        
        # 2) Check the channel counts that come out of the backbone
        backbone_info = self.backbone.feature_info
        # E.g. for resnet50: [64, 256, 512, 1024, 2048]
        self.filters = [backbone_info.channels(i) for i in range(5)]
        
        # We'll unify them to cat_channels = self.filters[0], but you can pick your own logic
        self.cat_channels = self.filters[0]  # typically 64 for ResNet50's first feature
        self.cat_blocks = len(self.filters)  # 5
        self.upsample_channels = self.cat_blocks * self.cat_channels  # e.g. 5*64 = 320

        # Classification Guided Module (if you still want it)
        if self.CGM:
            self.cgm = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(self.filters[-1], 2, kernel_size=1, padding=0),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Sigmoid()
            )
        else:
            self.cgm = None

        # 3) Define the decoder. The example below is simplistic. 
        #    We'll replicate your prior code's logic for d4, d3, d2, d1. 
        #    But you need to map: e1, e2, e3, e4, e5 => features[0..4]
        #    So for d4, for example:
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),  # e1 -> (64 -> 64)
            ConvBlock(self.filters[1], self.cat_channels, n=1),  # e2 -> (256->64)
            ConvBlock(self.filters[2], self.cat_channels, n=1),  # e3 -> (512->64)
            ConvBlock(self.filters[3], self.cat_channels, n=1),  # e4 -> (1024->64)
            ConvBlock(self.filters[4], self.cat_channels, n=1),  # e5 -> (2048->64)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        # d3, d2, d1 similarly: adjust as needed
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
        if self.deep_supervision:
            self.deep_sup = nn.ModuleList([
                ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False)
                for _ in range(3)
            ] + [
                ConvBlock(self.filters[-1], output_channels, n=1, is_bn=False, is_relu=False)
            ])
        else:
            self.deep_sup = None

    def forward(self, x):
        # 1) Extract features from backbone
        #    This will be a list of 5 feature maps, e.g. [e1, e2, e3, e4, e5]
        #    with shapes:
        #    e1 = [B, 64,   H/4,  W/4 ]
        #    e2 = [B, 256,  H/4,  W/4 ]
        #    e3 = [B, 512,  H/8,  W/8 ]
        #    e4 = [B, 1024, H/16, W/16]
        #    e5 = [B, 2048, H/32, W/32]
        features = self.backbone(x)
        e1, e2, e3, e4, e5 = features

        # Classification guided module
        if self.CGM:
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # Now replicate your UNet3+ decode logic using e1..e5

        # d4
        d4 = [
            F.max_pool2d(e1, 8),       # e1 -> /8
            F.max_pool2d(e2, 4),       # e2 -> /4
            F.max_pool2d(e3, 2),       # e3 -> /2
            e4,                        # e4
            F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = torch.cat(d4, dim=1)
        d4 = self.d4_conv(d4)

        # d3
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

        # d2
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

        # d1
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
        if self.deep_supervision and self.training:
            outputs.extend([
                F.interpolate(self.deep_sup[0](d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[1](d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[2](d4), scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[3](e5), scale_factor=16, mode='bilinear', align_corners=True)
            ])

        # CGM
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]

        # Final ReLU
        outputs = [F.relu(out) for out in outputs]

        if self.deep_supervision and self.training:
            return torch.cat(outputs, dim=0)
        else:
            return outputs[0]


def train_base_model(suffix, checkpoint=None): #encoder_name, encoder_weights, decoder_attention_type
    wandb.finish()    
    
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
    INPUT_SHAPE = [in_channels, 256, 256]
    OUTPUT_CHANNELS = 1
    encoder_name = ""

    model = UNet3PlusTIMM(backbone_name=encoder_name, OUTPUT_CHANNELS, deep_supervision=True, cgm=False, training=True)
    '''model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, 
                             decoder_attention_type=decoder_attention_type,
                             in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S'], classes=1, activation=None)'''

    if checkpoint!=None:
        s2s1_model = SentinelModel.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path, 
                                                    mean_agb=mean_agb, std_agb=std_agb,
                                                    lr=0.0005, wd=0.0001)
    else:
        s2s1_model = SentinelModel(model, mean_agb=mean_agb, std_agb=std_agb, lr=0.001, wd=0.0001)

    # summary(s2s1_model.cuda(), (S2_CHANNELS['6S']+S1_CHANNELS['6S'], 256, 256)) 

    wandb_logger = WandbLogger(save_dir=f'./models', name=f'{encoder_name}_6S', 
                               project=f'{encoder_name}_6S')

    # Define a trainer and start training:
    on_best_valid_loss = ModelCheckpoint(filename="{epoch}-{valid/loss}", mode='min', save_last=True,
                                         monitor='valid/loss', save_top_k=2)
    on_best_valid_rmse = ModelCheckpoint(filename="{epoch}-{valid/rmse}", mode='min', save_last=True,
                                         monitor='valid/rmse', save_top_k=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = [on_best_valid_loss, on_best_valid_rmse, lr_monitor]

    # Initialize a trainer
    trainer = Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=100, 
                      logger=[wandb_logger], 
                      callbacks=checkpoint_callback)
    # Train the model ⚡
    trainer.fit(s2s1_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train_finetuned_model(checkpoint_path, suffix):
    wandb.finish()

    train_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                       s1_path=f"{root_dir}/train_features_s1_6S",
                                       agb_path=f"{root_dir}/train_agbm", X=X_train, mean=mean['6S'], std=std['6S'], 
                                       mean_agb=mean_agb, std_agb=std_agb, transform=None)

    val_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                     s1_path=f"{root_dir}/train_features_s1_6S",
                                     agb_path=f"{root_dir}/train_agbm", X=X_val, mean=mean['6S'], std=std['6S'], 
                                     mean_agb=mean_agb, std_agb=std_agb, transform=None)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=16, num_workers=8, pin_memory=True)

    in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S']
    INPUT_SHAPE = [in_channels, 256, 256]
    OUTPUT_CHANNELS = 1
    encoder_name = ""

    model = UNet3PlusTIMM(backbone_name=encoder_name, OUTPUT_CHANNELS, deep_supervision=True, cgm=False, training=True)
    '''model = smp.UnetPlusPlus(encoder_name=encoder_name, decoder_attention_type=decoder_attention_type,
                             in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S'], classes=1, activation=None)'''

    freeze_encoder(model)

    s2s1_model = SentinelModel.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path, 
                                                    mean_agb=mean_agb, std_agb=std_agb,
                                                    lr=0.0005, wd=0.0001)

    # summary(s2s1_model.cuda(), (S2_CHANNELS['6S']+S1_CHANNELS['6S'], 256, 256)) 


    wandb_logger = WandbLogger(save_dir=f'./models', name=f'{encoder_name}_6S', 
                               project=f'{encoder_name}_6S')

    ## Define a trainer and start training:
    on_best_valid_loss = ModelCheckpoint(filename="{epoch}-{valid/loss}", mode='min', save_last=True,
                                         monitor='valid/loss', save_top_k=2)
    on_best_valid_rmse = ModelCheckpoint(filename="{epoch}-{valid/rmse}", mode='min', save_last=True,
                                         monitor='valid/rmse', save_top_k=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = [on_best_valid_loss, on_best_valid_rmse, lr_monitor]

    # Initialize a trainer
    trainer = Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=50, 
                      # logger=[wandb_logger], 
                      callbacks=checkpoint_callback)
    # Train the model ⚡
    trainer.fit(s2s1_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train_base_model('6S')



if __name__ == '__main__':
    checkpoint_path = r'./models/se_resnext50_32x4d_6S/qji032p2/checkpoints/loss=0.07499314099550247.ckpt'
    train_finetuned_model(checkpoint_path, '6S')
