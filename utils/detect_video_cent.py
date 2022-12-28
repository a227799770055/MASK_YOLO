from copy import deepcopy
from genericpath import isdir
import sys
from centroidtracker import CentroidTrackers    #1123

sys.path.append('.')
import torch
import cv2
import numpy as np 
from model.od.data.datasets import letterbox
from typing import Any
from model.backbone_YOLO import *
from model.head_RCNN import *
from model.groundtrue_import import *
from model.utils.modelpacking import *
import time
import argparse
from sort.sort import *
from torchvision import transforms
import traceback
import matplotlib
import matplotlib.pyplot as plt

class VideoDetector:
    def __init__(self, config):
        self.cfg = config
        self.toTRT = config['totrt']
        self.device = 'cuda'
        self.yolo = self.loading_yolo_model()
        self.maskrcnn = self.loading_maskrcnn_model()
        self.img_shape = 640

    def loading_yolo_model(self):
        if not self.toTRT:
            yolo_wts = self.cfg['model']['weight']
            model = model_manipulate(yolo_wts).eval().to(self.device)
            return model
        elif self.toTRT:
            yolo_backbone_wts = self.cfg['model']['backbone']
            yolo_backbone = loadTRTmodel(yolo_backbone_wts)
            yolo_head_wts = self.cfg['model']['headdetector']
            yolo_head = torch.load(yolo_head_wts)
            model = yoloModelPack2TRT(yolo_backbone, yolo_head)
            return model       

    def loading_maskrcnn_model(self):
        if not self.toTRT:
            maskrcnn_wts = self.cfg['maskrcnn']['weight']
            model = torch.load(maskrcnn_wts).eval().to(self.device)
            return model
        elif self.toTRT:
            mask_featurealign_wts = self.cfg['maskrcnn']['featurealign']
            mask_featurealign = loadTRTmodel(mask_featurealign_wts)
            mask_roipool_wts = self.cfg['maskrcnn']['roipool']
            mask_roipool = torch.load(mask_roipool_wts)
            mask_headdetector_wts = self.cfg['maskrcnn']['headdetector']
            mask_headdetector = loadTRTmodel(mask_headdetector_wts)
            model = maskModelPack2TRT(mask_featurealign, mask_roipool, mask_headdetector)
            return model

    def image_loading(self, frame):
        image = frame
        im0s =  deepcopy(image)
        img_h, img_w = image.shape[0], image.shape[1]
        image = letterbox(image, new_shape=self.img_shape)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        image = image.float()
        image /= 255.0
        return image, im0s, img_h, img_w
    
    def bbox_processing(self, rois, img, im0s, xcycwh=False):
        bboxes = []
        scores = []
        ids = []
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

    def mask_processing(self, mask, boxes):
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
    
    def frame_detect(self, frame):
        #   將image轉換成要進行推論的格式
        image, im0s, img_h, img_w = self.image_loading(frame)
        
        #   進行 object detection 推論
        pred = self.yolo(image)
        rois = non_max_suppression(pred['rois'][0],self.cfg['nms']['conf_thres'], self.cfg['nms']['iou_thres'],\
                                classes= self.cfg['nms']['classes'],agnostic=self.cfg['nms']['agnostic_nms'])
        boxes = rois[0][:,:4]
        f1,f2,f3 = pred['feature_map'][0], pred['feature_map'][1],pred['feature_map'][2]
        #   進行 segmenation 推論
        if len(boxes)!=0:
            mask = self.maskrcnn(f1,f2,f3, boxes)
        else:
            mask =[]
        #   box 和 mask 需要預處理成原圖格式
        #   box 預處理
        boxes, scores, ids = self.bbox_processing(rois, image, im0s)
        #   mask 預處理
        masks = self.mask_processing(mask,boxes)
        return boxes, scores, ids, masks

def  calculateOffset(bbs_first, bbs_second):
    # 計算前一個 bbs 與當前 bbs 的 offset
    #計算bbs_first 對角線距離
    #如果 offset > 對角線距離的 1/10 則使用新的 bbs
    xc0 = (bbs_first[0] + bbs_first[2])/2
    yc0 = (bbs_first[1] + bbs_first[3])/2
    xc1 = (bbs_second[0] + bbs_second[2])/2
    yc1 = (bbs_second[1] + bbs_second[3])/2
    # 對角線距離
    diagonal = ((bbs_first[0] - bbs_first[2])**2) + ((bbs_first[1] - bbs_first[3])**2)
    diagonal = diagonal**0.5
    #offset
    xc_off = xc1 - xc0
    yc_off = yc1 - yc0
    diagonal_off = ((xc_off**2) + (yc_off**2))**0.5
    #判定是否超出 threshold
    threshold = diagonal * 0.1
    if diagonal_off > threshold:
        new_bbs_x0 = (bbs_first[0] + bbs_second[0])/2
        new_bbs_y0 = (bbs_first[1] + bbs_second[1])/2
        new_bbs_x1 = (bbs_first[2] + bbs_second[2])/2
        new_bbs_y1 = (bbs_first[3] + bbs_second[3])/2
        return [new_bbs_x0, new_bbs_y0, new_bbs_x1, new_bbs_y1]
    else:
        return bbs_first

def mask_ploting(mask, framecopy, boxes):
    bg = np.zeros((framecopy.shape[0], framecopy.shape[1],3), np.uint8)
    bg = framecopy
    maskcopy = deepcopy(mask)
    mask = mask*255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #   橢圓擬合
    _,contours, hierarchy = cv2.findContours(maskcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

def video_detect(video_path, save_path, config):
    #   video informations
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_name = (video_path.split('/')[-1]).split('.')[0]
    out = cv2.VideoWriter(os.path.join(save_path, '{}.avi'.format(video_name)), fourcc, fps, (width,height))
    out2 = cv2.VideoWriter(os.path.join(save_path, '{}_mask.avi'.format(video_name)), fourcc, fps, (width,height))
    frameID = 0
    temp_id = {}
    temp_bbs = []

    Detector = VideoDetector(config=config)
    cen = CentroidTrackers() #1123
    while cap.isOpened():
        try:
            start = time.time()
            ret, frame = cap.read()
            framecopy = frame.copy()
            mask_bg = np.zeros((framecopy.shape[0], framecopy.shape[1],3), np.uint8)
            bboxes, scores, ids, masks = Detector.frame_detect(frame)
            rects = []
            for idx in range(bboxes.shape[0]):
                        bbox = bboxes[idx].astype(int)
                        score = round(scores[idx],2)
                        rect = []
                        for i in bbox:
                            rect.append(i)
                        # rect.append(score)
                        rects.append(rect)
            
            objects = cen.update(rects) #1123
            if len(rects) != 0: #1123 下面縮排
                for objectID, center in objects.items():
                    x0 = int(center[2])
                    y0 = int(center[3])
                    x1 = int(center[4])
                    y1 = int(center[5])
                    objectID = objectID
                    if objectID not in temp_id:
                        temp_id[objectID] = 0
                    elif objectID in temp_id and temp_id[objectID]%2 == 0:
                        temp_id.update({objectID:temp_id[objectID]+1})
                    else:
                        temp_id.update({objectID:temp_id[objectID]+1})
                    if temp_id[objectID] == 0:
                        temp_bbs = [x0, y0, x1, y1]
                        cv2.rectangle(framecopy, (int(temp_bbs[0]), int(temp_bbs[1])), (int(temp_bbs[2]), int(temp_bbs[3])), (0, 255, 0), 3, 1)
                        bboxes = [temp_bbs]
                        # mask_bg = mask_ploting(masks, framecopy, [temp_bbs]) 
                        cv2.putText(framecopy, str(objectID), (int((x1+x0)/2),int((y1+y0)/2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255,0,0), 2, cv2.LINE_AA)
                    elif temp_id[objectID] > 0:
                        temp_bbs = calculateOffset(temp_bbs, center[2:6])
                        cv2.rectangle(framecopy, (int(temp_bbs[0]), int(temp_bbs[1])), (int(temp_bbs[2]), int(temp_bbs[3])), (0, 255, 0), 3, 1)
                        bboxes = [temp_bbs]
                        # mask_bg = mask_ploting(masks, framecopy, [temp_bbs])
                        cv2.putText(framecopy, str(objectID), (int((x1+x0)/2),int((y1+y0)/2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255,0,0), 2, cv2.LINE_AA)

            end = time.time()
            frameID += 1
            out.write(framecopy)
            out2.write(mask_bg)
            cv2.imshow('frame', framecopy)
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            error_class = e.__class__.__name__ #取得錯誤類型
            detail = e.args[0] #取得詳細內容
            cl, exc, tb = sys.exc_info() #取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
            fileName = lastCallStack[0] #取得發生的檔案名稱
            lineNum = lastCallStack[1] #取得發生的行號
            funcName = lastCallStack[2] #取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print('Finish...')

def parse_opt():
    parser = argparse.ArgumentParser(
        prog="detector_video_cent.py",
    )
    parser.add_argument(
        '--cfgPath',
        type=str,
        default='config/config.yaml',
        help='path of the config.'
    )
    parser.add_argument(
        '--videoPath',
        type=str,
        default='/home/insign/影片/SORT_Test',
        help='path of the video.'
    )
    parser.add_argument(
        '--savePath',
        type=str,
        default='/home/insign/影片/SORT_Result',
        help='path of the result.'  
    )
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    #   Input parse opt
    opt = parse_opt()
    videoPath = opt.videoPath
    cfgPath = opt.cfgPath
    savePath = opt.savePath

    #   Loading config file
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    #   List videos
    videos = os.listdir(videoPath)
    for video in videos:
        videoP = os.path.join(videoPath, video)
        saveP = os.path.join(savePath, video.split('.')[0])
        if not os.path.isdir(saveP):
            os.mkdir(saveP)
        #   video inference start
        video_detect(videoP, saveP, cfg)