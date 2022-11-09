from copy import deepcopy
import sys, os, time
sys.path.append('./model')
import torch
import cv2
import numpy as np 
from od.data.datasets import letterbox
from typing import Any
from backbone_YOLO import *
from head_RCNN import *
from groundtrue_import import *
from PIL import Image
from torchvision import transforms

## Depth package
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from depth.apis import multi_gpu_test, single_gpu_test
from depth.datasets import build_dataloader, build_dataset
from depth.models import build_depther

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

def merge_mask_image(mask, im0s, retval):
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    det = im0s[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)]
    retval_bg = np.zeros((det.shape[0], det.shape[1],3), np.uint8)
    for i in range(len(retval)):
        retval_bg = cv2.ellipse(retval_bg, retval[i], (0, 0, 255), thickness=-1)
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
    f1,f2,f3 = pred['feature_map'][0], pred['feature_map'][1],pred['feature_map'][2]
    cv2.rectangle(im0s, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)

    #   Resize to bounding box
    h = int(boxes[0][2] - boxes[0][0])
    w = int(boxes[0][3] - boxes[0][1])
    resize_to_bbs = transforms.Compose([transforms.Resize((w, h))])
    
    #   Predict mask
    mask_logits = mask_head( f1,f2,f3, boxes)
    mask_logits = resize_to_bbs(mask_logits)
    mask_logits = mask_logits.detach().cpu().numpy()
    mask_logits = mask_logits[0][0]>0.6
    mask_logits.dtype = 'uint8'
    mask = deepcopy(mask_logits)
    mask_logits = mask_logits*255
    mask_logits = cv2.cvtColor(mask_logits, cv2.COLOR_GRAY2RGB)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retval = []
    for i in range(len(contours)):
        if contours[i].shape[0]<5: # filter the number of points under five
            continue
        ellipse = cv2.fitEllipse(contours[i])
        ellipse, ellipse[1] = list(ellipse), list(ellipse[1])
        ellipse[1][0], ellipse[1][1] = ellipse[1][0]*1, ellipse[1][1]*1
        if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
            continue
        retval.append(ellipse)
    return boxes, mask_logits, retval

# dpeth
def depth_estimation(image, model, depth_cfg):
    origin_w, origin_h = image.shape[3], image.shape[2]
    resize_to_512 = transforms.Compose([transforms.Resize((512, 512))])
    with torch.no_grad():
        input_img = resize_to_512(image)

        result = model(img = [input_img])[0][0]
        result = cv2.resize(result, dsize=(origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    return result

def load_depth_cfg(path):
    cfg = mmcv.Config.fromfile(path)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    return cfg

def load_depth_model(cfg, pth_path):
    model = build_depther(
        cfg.model,
        test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, pth_path)
    model.eval()
    return model

def getFake():
    meta = dict()
    meta['filename'] = 'test.png'
    meta['ori_filename'] = 'test.png'
    meta['ori_shape'] = (512, 512, 3)
    meta['img_shape'] = (512, 512, 3)
    meta['pad_shape'] = (512, 512, 3)
    meta['scale_factor'] = [1., 1., 1., 1.]
    meta['flip'] = False
    meta['flip_direction'] = 'horizontal'
    meta['to_rgb'] = True
    return meta

def evalDepthPose(depth, px, py):
    for i in range(10):
        depth[py + i - 5][px + i - 5] = 0.
        depth[py + i - 5][px - i + 5] = 0.

def plotRealTargetSize(img, boxes, depth, FOV_W=140, FOV_H=140, decimal = 2):
    x = int(boxes[0][0])
    y = int(boxes[0][1])
    h = int(boxes[0][3] - boxes[0][1])
    w = int(boxes[0][2] - boxes[0][0])
    text_color = (0, 0, 0)
    px = int(x + (w / 2))
    py = int(y + (h / 2))
    img = cv2.circle(img, (px,py), radius=5, color=(255, 0, 0), thickness=-1)
    # d
    target_d = depth[py][px]

    # evalDepthPose(depth, px, py)

    text = "d:" + str(round(target_d, decimal))  + 'cm'
    img = cv2.putText(img, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    # w, h
    img_w = img.shape[0]
    img_h = img.shape[1]
    
    theta_x = (w / 2 /img_w) * (FOV_W / 180.) * np.pi
    theta_y = (h / 2 /img_h) * (FOV_H / 180.) * np.pi
    target_w = target_d * np.tanh(theta_x) * 2.
    target_h = target_d * np.tanh(theta_y) * 2.

    # msg
    shiftPix = 30
    text = 'h:' + str(round(target_h, decimal))  + 'cm'
    img = cv2.putText(img, text, (px, py + shiftPix), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    text = "w:" + str(round(target_w, decimal))  + 'cm'
    img = cv2.putText(img, text, (px, py + (shiftPix * 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    return img
    
if __name__ == '__main__':
    imgDir = 'data/Validate'
    cfgPath = 'config/config.yaml'
    
    #   Loading cfg
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    #   Load depth cfg
    depth_cfg = load_depth_cfg(cfg['depth']['config'])

    roimodelPath = cfg['maskrcnn']['weight']

    save_dir = 'result'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    device = 'cuda'

    #   Model yolo_backbone loading
    yolo = model_manipulate(cfg['model']['weight']).eval().to(device)
    #   Model roi_head loading
    mask_head = torch.load(roimodelPath).eval().to(device)
    #   Model depth swint load weight
    depthformer_net = load_depth_model(depth_cfg, cfg['depth']['weight']).to(device)

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
        depth_val = depth_estimation(image, depthformer_net, depth_cfg)
        if len(boxes) != 0:
            #   Merge mask and image
            im0s = merge_mask_image(mask=mask_logits, im0s=im0s, retval=retval)
            im0s = plotRealTargetSize(im0s, boxes, depth_val, cfg['depth']['FOV_W'], cfg['depth']['FOV_H'])
            cv2.imwrite('{}/{}_det.jpg'.format(save_dir, name),im0s)
            
            # output depth image
            depth_show = (depth_val / depth_val.max() * 255.).astype(np.uint8)
            depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)
            depth_show = merge_mask_image(mask=mask_logits, im0s=depth_show, retval=retval)
            depth_show = plotRealTargetSize(depth_show, boxes, depth_val, cfg['depth']['FOV_W'], cfg['depth']['FOV_H'])
            cv2.imwrite('{}/{}_dpeth.jpg'.format(save_dir, name), depth_show)
        else:
            print("-"*15)
            print("Do not detect polyp")
        
        

        e = time.time()
        print('Total Time Duration = {} ms'.format((e-s)*1000))