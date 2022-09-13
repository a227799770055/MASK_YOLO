import os 
import cv2
import torch
import numpy as np
import json
from torchvision.ops import roi_align
import math

def convertPredList2Tensor(pred):    
    assert type(pred) == dict
    keys = pred.keys()
    for key in keys:
        lenList = len(pred[key])
        for i in range(lenList):
            pred[key][i] = torch.FloatTensor( pred[key][i])

def maskManipulate(mask_path):
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    #   resize mask to [480 480]
    mask0 = mask.copy()
    mask = cv2.resize(mask, (480,480), interpolation=cv2.INTER_LINEAR)
    mask = torch.from_numpy(mask).to('cuda')
    mask = mask.float()
    mask /= 255.0
    mask = torch.ceil(mask).float()
    mask = mask.unsqueeze(0).unsqueeze(0) 
    return mask

def maskRoiAlign(mask, boxes, output_size):
    mask_roi = roi_align(mask, [boxes], (output_size,output_size), 1.0)
    return mask_roi

if __name__ == '__main__':
    mask_path = '/home/insign/Doc/insign/Mask_yolo/Image/Mask/2022012001_33_m.jpg'
    mask = maskManipulate(mask_path)
    # loading boxes
    jsonDir = '/home/insign/Doc/insign/flexible-yolov5/json_save.json'
    with open(jsonDir, 'r') as f:
        predJSON = json.load(f)
    #   convert list to tensor
    convertPredList2Tensor(predJSON)
    boxes = predJSON['rois'][0][:,:4].to('cuda')
    #   roi align
    mask_roi =  maskRoiAlign(mask, boxes)




