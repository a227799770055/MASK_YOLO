import sys
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule
import torch
import datetime, time, copy, yaml
device = torch.device('cuda')
from copy import deepcopy
import sys, os, time
sys.path.append('/home/insign/Doc/insign/Mask_yolo')
import torch
import cv2
import numpy as np 
from model.od.data.datasets import letterbox
from typing import Any
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.groundtrue_import import *
from PIL import Image
from torchvision import transforms
from model.utils.modelpacking import *

class FeatureAlign(nn.Module):
    def __init__(self, featalign1, featalign2, featalign3):
        super(FeatureAlign, self).__init__()
        self.featalign1 = featalign1
        self.featalign2 = featalign2
        self.featalign3 = featalign3
    def forward(self, feat1, feat2, feat3):
        features_1 = self.featalign1(feat1)
        features_2 = self.featalign2(feat2)
        features_3 = self.featalign3(feat3)
        return features_1, features_2, features_3

class HeadDetector(nn.Module):
    def __init__(self, head, detector):
        super(HeadDetector, self).__init__()
        self.head = head
        self.detector = detector
    def forward(self, feat):
        mask = self.head(feat)
        mask = self.detector(mask)
        return mask



#%%
toTRT = True
#%%
weight_path = '/home/insign/Doc/insign/Mask_yolo/run/0921/best.pt'
model = torch.load(weight_path).eval().to('cuda')
print(model)

feat1 = torch.rand((1,256,80,80)).to('cuda')
feat2 = torch.rand((1,512,40,40)).to('cuda')
feat3 = torch.rand((1,1024,20,20)).to('cuda')
boxes = torch.rand((1,4)).to('cuda')

# Feature align 2 TRT
featalign1 = model.featurealign1
featalign2 = model.featurealign2
featalign3 = model.featurealign3
featurealign = FeatureAlign(featalign1, featalign2, featalign3)

f1, f2, f3 = featurealign(feat1, feat2, feat3)
if toTRT:
    model_trt = torch2trt(featurealign, [feat1, feat2, feat3], int8_mode=True)
    torch.save(model_trt.state_dict(), 'toTRT/featurealign.pth')
featurealignTRT = loadTRTmodel('toTRT/featurealign.pth')

# Feature Packing
features = {}
key_name = ["feat1","feat2","feat3"]
feature_map = [f1,f2,f3]
for i,j in zip(key_name, feature_map):
        features[i] = j

# ROI pooling
image_shapes = [(640,640)]
roipool = model.mask_roi_pool
torch.save(roipool, 'toTRT/roipool.pth')
# featpool = roipool(features, [boxes], image_shapes) # output shape = (1, 512, 28, 28)

# Mask Head & Detector
featpool = torch.rand((32, 512, 28, 28)).to('cuda')

maskhead = model.mask_head
detector = model.fcn_predictor
headdetector = HeadDetector(maskhead, detector)
mask = headdetector(featpool)

if toTRT:
    model_trt = torch2trt(headdetector, [featpool], int8_mode=True)
    torch.save(model_trt.state_dict(), 'toTRT/headdetector.pth')


# Test model packing
