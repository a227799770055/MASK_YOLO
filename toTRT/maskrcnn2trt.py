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
import argparse


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


def parse_opt():
    parser = argparse.ArgumentParser(
        prog="maskrcnn2trt.py",
    )
    parser.add_argument(
        '--maskPath',
        type=str,
        default='/home/insign/Doc/insign/Mask_yolo/run/0921/best.pt',
        help='path of the config.'
    )
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    toTRT = True
    opt = parse_opt()

    weight_path = opt.maskPath
    model = torch.load(weight_path).eval().to('cuda')

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
        print('start convert featurealign pytorch to tensorRT')
        featurealign_trt = torch2trt(featurealign, [feat1, feat2, feat3], int8_mode=True)
        torch.save(featurealign_trt.state_dict(), 'toTRT/featurealign.pth')

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

    # Mask Head & Detector
    featpool = torch.rand((32, 512, 28, 28)).to('cuda')
    maskhead = model.mask_head
    detector = model.fcn_predictor
    headdetector = HeadDetector(maskhead, detector)
    mask = headdetector(featpool)
    if toTRT:
        print('start convert headdetector pytorch to tensorRT')
        headdetector_trt = torch2trt(headdetector, [featpool], int8_mode=True)
        torch.save(headdetector_trt.state_dict(), 'toTRT/headdetector.pth')

    # inference time test
    trtmodel = maskModelPack2TRT(featurealign_trt, roipool, headdetector_trt)
    for i in range(100):
        s = time.time()
        y = trtmodel(feat1, feat2, feat3, boxes)
        e = time.time()
        print('Inference time take {} ms'.format((e-s)*1000))

