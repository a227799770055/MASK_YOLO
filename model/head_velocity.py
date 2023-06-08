import torch
from torch import nn 
import os
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import json

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)

class VelocityHeads(nn.Module):
    def __init__(
        self,
        out_channels=6,
         ):
        super().__init__()

        self.featurealign1 = nn.Conv2d(256, 512, 1, 1).to('cuda')
        self.featurealign2 = nn.Conv2d(512, 512, 1, 1).to('cuda')
        self.featurealign3 = nn.Conv2d(1024, 512, 1, 1).to('cuda')
        
        self.fcn1 = FCNHead(512, 256)
        self.fcn2 = FCNHead(256, 128)
        self.fcn3 = FCNHead(128, 64)
        self.fc = nn.Linear(64 * 6, out_channels)
        
    def forward_fcn(self, x):
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        return x
    
    def forward(
        self,
        features_11,
        features_12,
        features_13,
        features_21,
        features_22,
        features_23,
        image_shapes=[(480,480)],
        targets=None):

        batch_size = features_11.size(0)
        features_11 = self.featurealign1(features_11)
        features_12 = self.featurealign2(features_12)
        features_13 = self.featurealign3(features_13)
        
        features_21 = self.featurealign1(features_21)
        features_22 = self.featurealign2(features_22)
        features_23 = self.featurealign3(features_23)
        
        f_11 = self.forward_fcn(features_11)
        f_12 = self.forward_fcn(features_12)
        f_13 = self.forward_fcn(features_13)
         
        f_21 = self.forward_fcn(features_21)
        f_22 = self.forward_fcn(features_22)
        f_23 = self.forward_fcn(features_23)
        
        # max pooling
        f_11 = nn.MaxPool2d(60, 60)(f_11).view(batch_size, -1)
        f_12 = nn.MaxPool2d(30,30)(f_12).view(batch_size, -1)
        f_13 = nn.MaxPool2d(15,15)(f_13).view(batch_size, -1)
        f_21 = nn.MaxPool2d(60,60)(f_21).view(batch_size, -1)
        f_22 = nn.MaxPool2d(30,30)(f_22).view(batch_size, -1)
        f_23 = nn.MaxPool2d(15,15)(f_23).view(batch_size, -1)
        
        # cat 
        features = torch.cat((f_11, f_12, f_13, f_21, f_22, f_23), dim = 1)
        pose = self.fc(features)
        
        # print("features_11", features_11.size())
        # print("features_12", features_12.size())
        # print("features_13", features_13.size())
        # print("features_21", features_21.size())
        # print("features_22", features_22.size())
        # print("features_23", features_23.size())
        # print("f_11", f_11.size())
        # print("f_12", f_12.size())
        # print("f_13", f_13.size())
        # print("f_21", f_21.size())
        # print("f_22", f_22.size())
        # print("f_23", f_23.size())
        # print("features", features.size())
        
        return pose

            

def convertPredList2Tensor(pred):    
    assert type(pred) == dict
    keys = pred.keys()
    for key in keys:
        lenList = len(pred[key])
        for i in range(lenList):
            pred[key][i] = torch.FloatTensor( pred[key][i])

# def featuremapPack(feature_map):
#     from collections import OrderedDict
#     map = OrderedDict()
#     key_name = ["feat1","feat2","feat3"]
#     for i,j in zip(key_name, feature_map):
#         map[i] = j
#     return map

def featuremapPack(feature_map):
    map = {}
    key_name = ["feat1","feat2","feat3"]
    for i,j in zip(key_name, feature_map):
        map[i] = j
    return map

if __name__ == '__main__':
    
    feat11 = torch.randn(1,512,60,60)
    feat12 = torch.randn(1,512,30,30)
    feat13 = torch.randn(1,512,15,15)
    
    feat21 = torch.randn(1,512,60,60)
    feat22 = torch.randn(1,512,30,30)
    feat23 = torch.randn(1,512,15,15)
    
    feat = {
        "feat11": feat11,
        "feat12": feat12,
        "feat13": feat13,
        "feat21": feat21,
        "feat22": feat22,
        "feat23": feat23
    }
    
    veloHead = VelocityHeads()
    
    y = veloHead(feat11, feat12, feat13, feat21, feat22, feat23)
    print(y.size())