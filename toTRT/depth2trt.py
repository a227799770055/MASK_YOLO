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
sys.path.append('/home/insign/Doc/insign/Mask_yolo')
from model.od.data.datasets import letterbox
from typing import Any
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.groundtrue_import import *
import argparse
import mmcv
from mmcv.runner import load_checkpoint
from depth.models import build_depther
from torchvision import transforms

def load_depth_model(cfg, pth_path):
    model = build_depther(
        cfg.model,
        test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, pth_path)
    return model

def depth_estimation(image, model):
    # model = MMDataParallel(model, device_ids=[0])
    ori_h, ori_w =  image.shape[2], image.shape[3]
    model.eval()
    resize_to_512 = transforms.Compose([transforms.Resize((512, 512))])
    with torch.no_grad():
        input_img = resize_to_512(image)
        
        result = model(img = [input_img])[0][0]
        result = cv2.resize(result, dsize=(ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    return result

def load_depth_cfg(path):
    cfg = mmcv.Config.fromfile(path)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    return cfg

if __name__ == '__main__':
    toTRT = True
    depth_cfg=load_depth_cfg('./data/depthformer_swint_w7_endoscopy.py')
    depth_cfg2='./data/depth_best_weight.pth'
    depthformer_net = load_depth_model(depth_cfg, depth_cfg2).to(device)
    
    x = torch.ones((64,3,512,512)).cuda()

    print(x)
    #transfer torch to trt weight and save as pth
    if toTRT:
        print('start convert pytorch to tensorRT')
        model_trt = torch2trt(depthformer_net, [x], fp16_mode=True)
        torch.save(model_trt.state_dict(), 'toTRT/depth.pth')
        

    x = torch.rand((1,3,512,512)).cuda()

    print('test depth inference time')
    
    for i in range(10):
        s = time.time()
        y = depthformer_net(x)
        e = time.time()
        print('Inference time take {} ms'.format((e-s)*1000))
