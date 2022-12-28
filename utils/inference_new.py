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

def mask_ploting(mask, framecopy, boxes):
    bg = np.zeros((framecopy.shape[0], framecopy.shape[1],3), np.uint8)
    bg = framecopy
    maskcopy = deepcopy(mask)
    mask = mask*255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #   橢圓擬合
    contours, hierarchy = cv2.findContours(maskcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retval = []
    for i in range(len(contours)):
        if contours[i].shape[0]<5: # filter the number of points under five
            continue
        ellipse = cv2.fitEllipse(contours[i])
        ellipse, ellipse[1] = list(ellipse), list(ellipse[1])
        ellipse[1][0], ellipse[1][1] = ellipse[1][0], ellipse[1][1]
        if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
            continue
        retval.append(ellipse)
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    boxes[0][1], boxes[0][0] = max(0, boxes[0][1]), max(0, boxes[0][0])
    det = bg[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)]
    retval_bg = np.zeros((mask_h, mask_w,3), np.uint8)
    for i in range(len(retval)):
        retval_bg = cv2.ellipse(retval_bg, retval[i], (255, 255, 255), thickness=-1)
    # 將橢圓和原本的 mask 融合
    mask = mask+retval_bg
    mask = mask[0:det.shape[0], 0:det.shape[1]]
    det = cv2.addWeighted(det,0.3 ,mask, 0.7, 0)
    bg[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)] = det
    return bg

def bbox_processing(rois, img, im0s, xcycwh=False):
    bboxes = []
    scores = []
    ids = []
    h_ratio = im0s.shape[0]/img.shape[2]
    w_ratio = im0s.shape[1]/img.shape[3]
    for i, det in enumerate(rois):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in det:
                x_min = xyxy[0].cpu() 
                y_min = xyxy[1].cpu() 
                x_max = xyxy[2].cpu() 
                y_max = xyxy[3].cpu() 
                score = conf.cpu()
                clas = cls.cpu()
                w = x_max - x_min
                h = y_max - y_min
                if xcycwh:
                    # center coord, w, h
                    bboxes.append([x_min + w / 2, y_min + h / 2, w, h])
                else:
                    bboxes.append([x_min, y_min, x_max, y_max])
                scores.append(score)
                ids.append(clas)
    return np.asarray(bboxes), np.asarray(scores), np.asarray(ids)

def mask_processing(mask, boxes):
    if len(mask) != 0:
        # 將尺寸調回去 bbox 大小 並轉成 binary mask
        h = int(boxes[0][2] - boxes[0][0])
        w = int(boxes[0][3] - boxes[0][1])
        resize_to_bbs = transforms.Compose([transforms.Resize((w, h))])
        mask = resize_to_bbs(mask)
        # 由 tensor --> numpy 格式
        mask = mask.detach().cpu().numpy()
        mask = mask[0][0]>0.7
        mask.dtype = 'uint8'
    return mask

def image_loading(img_path):
    image = cv2.imread(str(img_path))
    img_h, img_w = image.shape[0], image.shape[1]
    im0s = deepcopy(image)
    image = letterbox(image, new_shape=640)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    image = image.float()
    image /= 255.0

    #   image input for depth estimate
    imageDepthInput = deepcopy(im0s)
    imageDepthInput = imageDepthInput[:, :, ::-1].transpose(2, 0, 1)
    imageDepthInput = np.ascontiguousarray(imageDepthInput)
    imageDepthInput = torch.from_numpy(imageDepthInput).unsqueeze(0).to(device)
    imageDepthInput = imageDepthInput.float()
    imageDepthInput /= 255.0

    return image, im0s, img_h, img_w, imageDepthInput

def model_detection(image, yolo, mask_head, cfg):
    #   進行 object detection inference
    pred = yolo(image)
    rois = non_max_suppression(pred['rois'][0],cfg['nms']['conf_thres'], cfg['nms']['iou_thres'], \
                                    classes= cfg['nms']['classes'],agnostic=cfg['nms']['agnostic_nms'])
    boxes = rois[0][:,:4]#  rois
    feature_map = featuremapPack(pred['feature_map']) #   extract feature map and boxes
    f1,f2,f3 = pred['feature_map'][0], pred['feature_map'][1],pred['feature_map'][2]
    #   進行 segmentation inference
    if len(boxes) != 0:
        mask = mask_head( f1,f2,f3, boxes)
    else:
        mask = []
    
    #   box 和 mask 需要預處理成原圖格式
    #   box 預處理
    boxes, scores, ids = bbox_processing(rois, image, im0s)
    print(len(boxes))
    if len(boxes!=0):
        cv2.rectangle(im0s, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)
    #   mask 預處理
    masks = mask_processing(mask,boxes)

    return boxes, scores, ids, masks

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

def plotRealTargetSize(img, boxes, depth, FOV_W=140, FOV_H=140):
    x = int(boxes[0][0])
    y = int(boxes[0][1])
    h = int(boxes[0][3] - boxes[0][1])
    w = int(boxes[0][2] - boxes[0][0])
    text_color = (255, 0, 0)
    px = int(x + (w / 2))
    py = int(y + (h / 2))
    img = cv2.circle(img, (px+10,py), radius=5, color=(255, 0, 0), thickness=-1)
    # d
    target_d = depth[py][px]
    text = "d:" + str(round(target_d, 2))  + 'cm'
    img = cv2.putText(img, text, (px+30, py), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    # w, h
    img_w = img.shape[0]
    img_h = img.shape[1]
    
    theta_x = (w / 2 /img_w) * (FOV_W / 180.) * np.pi
    theta_y = (h / 2 /img_h) * (FOV_H / 180.) * np.pi
    target_w = target_d * np.tanh(theta_x) * 2.
    target_h = target_d * np.tanh(theta_y) * 2.
    text = "w:" + str(round(target_w, 2))  + 'cm' 
    img = cv2.putText(img, text, (px+30, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    text = 'h:' + str(round(target_h, 2))  + 'cm'
    img = cv2.putText(img, text, (px+30, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    return img
    
if __name__ == '__main__':
    imgDir = 'Polyp/Validate'
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
        image, im0s, img_h, img_w, imageDepthInput = image_loading(imgPath)
        #   prediction
        boxes, scores, ids, masks = model_detection(image, yolo, mask_head, cfg)
        depth_val = depth_estimation(imageDepthInput, depthformer_net, depth_cfg)
        # cv2.imwrite('{}/{}_dpeth.jpg'.format(save_dir, name), depth_val * 255.)
        if len(boxes) != 0:
            #   Mask ploting
            bg = mask_ploting(masks, im0s, boxes)
            im0s = plotRealTargetSize(im0s, boxes, depth_val)
            cv2.imwrite('{}/{}_det.jpg'.format(save_dir, name),im0s)
        else:
            print("-"*15)
            print("Do not detect polyp")
        e = time.time()
        print('Total Time Duration = {} ms'.format((e-s)*1000))