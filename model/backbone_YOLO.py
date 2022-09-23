import os 
import torch
import sys
sys.path.append('./model')
import cv2
from od.data.datasets import letterbox
import numpy as np
from model.utils.modelpacking import yoloModelPack
import yaml
from utils.general import *    
import json


# import model
def model_manipulate(model_weight):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_weight)['model'] 
    model_backbone = model.backbone
    model_fpn = model.fpn
    model_pan = model.pan
    model_head = model.detection
    model = yoloModelPack(backbone=model_backbone, fpn=model_fpn, pan=model_pan, detector=model_head)
    return model
    

# import img
def img_munipulate(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    im0s = cv2.imread(image_path)
    img = letterbox(im0s, new_shape=480)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
                img = img.unsqueeze(0)
    return img,im0s

def bbox_scale(im0s, rois, inputsize=[480,480] ):
    im0s_h = im0s.shape[0]
    im0s_w = im0s.shape[1]
    input_w = inputsize[0]
    input_h = inputsize[1]
    h_factor = input_h /im0s_h
    w_factor = input_w /im0s_w
    for i in rois:
        i[:,0] = i[:,0] * h_factor
        i[:,2] = i[:,2] * h_factor
        i[:,1] = i[:,1] * w_factor
        i[:,3] = i[:,3] * w_factor
    return rois

#   convert tensor to array
def convertToList(pred):
    feat_map = pred['feature_map']
    new_map = []
    for i in feat_map:    
        i = i.cpu().detach().numpy()
        new_map.append(i.tolist())
    pred['feature_map'] = new_map
    roi = pred['rois'][0]
    roi = [roi.cpu().detach().numpy().tolist()]
    pred['rois'] = roi
    return pred


#   loading cfg
def cfg_load(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg

if __name__ == '__main__':
    yaml_cfg = '/home/insign/Doc/insign/Mask_yolo/config/yolocfg.yaml'
    image_path = '/home/insign/Doc/insign/flexible-yolov5/images/img/2022012001_33.jpg'
    
    with open(yaml_cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model_manipulate(cfg['model']['weight']).eval()
    img,im0s = img_munipulate(image_path)
    
    pred = model(img)
    rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])

    # rois = bbox_scale(im0s, rois) # rescale to img size
    pred['rois'] = rois
    print(pred['rois'])
    
    #   convert tensor to array
    pred = convertToList(pred)
    saveJSONName = 'json_save.json'
    with open(saveJSONName, 'w') as f:
        json.dump(pred, f)