import sys
sys.path.append('/home/insign/Doc/insign/Mask_yolo')
from model.head_RCNN import *
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule
import torch
from model.backbone_YOLO import *
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
from model.head_RCNN_trttest import *
from model.groundtrue_import import *
from PIL import Image
from torchvision import transforms

def image_loading(img_path):
    image = cv2.imread(str(img_path))
    img_h, img_w = image.shape[0], image.shape[1]
    image = letterbox(image, new_shape=640)[0]
    im0s = deepcopy(image)
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    image = image.float()
    image /= 255.0
    return image, im0s, img_h, img_w

image, im0s, img_h, img_w = image_loading('/home/insign/Doc/insign/Mask_yolo/Polyp/Images/2022012001_84.jpg')

yolo_cfg = '/home/insign/Doc/insign/Mask_yolo/config/config.yaml'
with open(yolo_cfg, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
device = torch.device('cuda')

yolo = model_manipulate(cfg['model']['weight']).eval().to(device)
x = image
pred = yolo(x)
rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])
boxes = rois[0][:,:4]
print(boxes)
feature_map = pred['feature_map']

device = torch.device('cuda')
model = ROIHeadsMask().to(device)
# mask_logits = model(feature_map[0], feature_map[1], feature_map[2], boxes)
model_trt = torch2trt(model, [feature_map[0], feature_map[1], feature_map[2], boxes])