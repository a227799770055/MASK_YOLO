import sys
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule
import torch
import datetime, time, copy, yaml
device = torch.device('cuda')
from copy import deepcopy
import sys, os, time
sys.path.append('.')
import torch
import cv2
import numpy as np 
from model.od.data.datasets import letterbox
from typing import Any
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.groundtrue_import import *
import argparse

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

def parse_opt():
    parser = argparse.ArgumentParser(
        prog="yolo2trt.py",
    )
    parser.add_argument(
        '--yoloPath',
        type=str,
        default='/home/insign/Doc/insign/flexible-yolov5/Polyp/AI_box_0706_toTRT/weights/best.pt',
        help='path of the config.'
    )
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    toTRT = True
    opt = parse_opt()

    weight_path = opt.yoloPath

    model = torch.load(weight_path)['model']
    model_backbone = model.backbone
    model_fpn = model.fpn
    model_pan = model.pan
    model_head = model.detection
    x = torch.ones((64,3,320,320)).cuda()
    model_feature_map = model_concat(model_backbone, model_fpn, model_pan).eval().cuda()

    #transfer torch to trt weight and save as pth
    if toTRT:
        print('start convert pytorch to tensorRT')
        model_trt = torch2trt(model_feature_map, [x], int8_mode=True)
        torch.save(model_trt.state_dict(), 'toTRT/MorphYolo_backbone.pth')
        torch.save(model_head, 'toTRT/MorphYolo_head.pth')

    print('test inference time')
    trtmodel = yoloModelPack2TRT(model_trt, model_head)
    for i in range(10):
        s = time.time()
        y = trtmodel(x)
        e = time.time()
        print('Inference time take {} ms'.format((e-s)*1000))