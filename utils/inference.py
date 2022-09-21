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

def merge_mask_image(mask, im0s, name, retval):
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    det = im0s[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)]
    retval_bg = np.zeros((det.shape[0], det.shape[1],3), np.uint8)
    for i in range(len(retval)):
        retval_bg = cv2.ellipse(retval_bg, retval[i], (255, 255, 255), thickness=-1)
    # a = cv2.addWeighted(retval_bg,0.5 ,mask, 0.5, 0)
    mask = mask+retval_bg
    det = cv2.addWeighted(det,0.7 ,mask, 0.3, 0)
    
    im0s[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)] = det
    return im0s

def model_detection(image, yolo, mask_head, cfg):
    #   Predict bounding box
    pred = yolo(image)
    rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])
    boxes = rois[0][:,:4]#  rois
    if len(boxes) == 0:
        return [], [], []
    feature_map = featuremapPack(pred['feature_map']) #   extract feature map and boxes
    cv2.rectangle(im0s, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)

    #   Resize to bounding box
    h = int(boxes[0][2] - boxes[0][0])
    w = int(boxes[0][3] - boxes[0][1])
    resize_to_bbs = transforms.Compose([transforms.Resize((w, h))])
    
    #   Predict mask
    mask_logits = mask_head(feature_map, boxes)
    mask_logits = resize_to_bbs(mask_logits)
    mask_logits = mask_logits.detach().cpu().numpy()
    mask_logits = mask_logits[0][0]>0.5
    mask_logits.dtype = 'uint8'
    mask = deepcopy(mask_logits)
    mask_logits = mask_logits*255
    #   morphology
    # kernel = np.ones((5,5),np.uint8)
    # mask_logits =  cv2.erode(mask_logits,kernel,iterations = 3)
    # mask_logits =  cv2.dilate(mask_logits,kernel,iterations = 5)
    mask_logits = cv2.cvtColor(mask_logits, cv2.COLOR_GRAY2RGB)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retval = []
    for i in range(len(contours)):
        if contours[i].shape[0]<5: # filter the number of points under five
            continue
        ellipse = cv2.fitEllipse(contours[i])
        ellipse, ellipse[1] = list(ellipse), list(ellipse[1])
        ellipse[1][0], ellipse[1][1] = ellipse[1][0]*1, ellipse[1][1]*1
        retval.append(ellipse)
    return boxes, mask_logits, retval

    
if __name__ == '__main__':
    imgDir = 'Polyp/Validate'
    cfgPath = 'config/config.yaml'
    #   Loading cfg
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    roimodelPath = cfg['maskrcnn']['weight']

    save_dir = 'result'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    device = 'cuda'

    #   Model yolo_backbone loading
    yolo = model_manipulate(cfg['model']['weight']).eval().to(device)
    #   Model roi_head loading
    mask_head = torch.load(roimodelPath).eval().to(device)

    imgs = os.listdir(imgDir)
    for i in imgs:
        print(i)
        imgPath = os.path.join(imgDir, i)
        name = i.split('.')[0]
        s = time.time()
        #   Image loading
        image, im0s, img_h, img_w = image_loading(imgPath)

        #   prediction
        boxes, mask_logits, retval = model_detection(image, yolo, mask_head, cfg)
        if len(boxes) != 0:
            #   Merge mask and image
            im0s = merge_mask_image(mask=mask_logits, im0s=im0s, name=name, retval=retval)
            cv2.imwrite('{}/{}_det.jpg'.format(save_dir, name),im0s)
        else:
            print("-"*15)
            print("Do not detect polyp")
        e = time.time()
        print('Total Time Duration = {} ms'.format((e-s)*1000))