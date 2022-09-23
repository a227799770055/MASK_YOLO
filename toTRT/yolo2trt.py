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




#concat backbone + fan + pan
class model_concat(nn.Module):
    def __init__(self, model_backbone, model_fpn, model_pan):
        super(model_concat, self).__init__()
        self.model_backbone = model_backbone
        self.model_fpn = model_fpn
        self.model_pan = model_pan
    def forward(self, x):
        out = self.model_backbone(x)
        out = self.model_fpn(out)
        out = self.model_pan(out)
        return out

class yoloModelPack2TRT(nn.Module):
    def __init__(self, backbone, detector):
        super(yoloModelPack2TRT, self).__init__()
        self.backbone = backbone
        self.detector = detector
    
    def forward(self, x):
        res = {}
        feat_map= []
        out1 = self.backbone(x)
        out2 = self.detector(list(out1))
        
        feat_map.append(out1[0])
        feat_map.append(out1[1])
        feat_map.append(out1[2])
        res['feature_map'] = (feat_map)
        res['rois'] = out2
        return res

#%%
toTRT = False
#%%
weight_path = '/home/insign/Doc/insign/flexible-yolov5/Polyp/AI_box_0706_toTRT/weights/best.pt'

model = torch.load(weight_path)['model']
model_backbone = model.backbone
model_fpn = model.fpn
model_pan = model.pan
model_head = model.detection
x = torch.ones((1,3,320,320)).cuda()
model_feature_map = model_concat(model_backbone, model_fpn, model_pan).eval().cuda()

#transfer torch to trt weight and save as pth
if toTRT:
    model_trt = torch2trt(model_feature_map, [x], int8_mode=True)
    torch.save(model_trt.state_dict(), 'toTRT/MorphYolo_backbone.pth')
    torch.save(model_head, 'toTRT/MorphYolo_head.pth')

backbone = TRTModule()
backbone.load_state_dict(torch.load('/home/insign/Doc/insign/Mask_yolo/toTRT/MorphYolo_backbone.pth'))
head = torch.load('/home/insign/Doc/insign/Mask_yolo/toTRT/MorphYolo_head.pth')

modelNew = yoloModelPack2TRT(backbone, head)
y = modelNew(x)