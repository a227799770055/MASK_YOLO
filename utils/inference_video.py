from copy import deepcopy
from genericpath import isdir
import sys
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
import traceback
import time

class VideoDetection:
    def __init__(self, maskrcnn_wts, cfg):
        self.cfg = cfg
        self.device = 'cuda'
        self.yolo = self.loading_yolo_model()
        self.maskrcnn_wts = maskrcnn_wts
        self.maskrcnn = self.loading_maskrcnn_model()

    def loading_yolo_model(self):
        yolo_wts = self.cfg['model']['weight']
        return model_manipulate(yolo_wts).eval().to(self.device)

    def loading_maskrcnn_model(self):
        maskrcnn = torch.load(self.maskrcnn_wts).eval().to(self.device)
        return maskrcnn

    def image_loading(self, frame):
        image = frame
        img_h, img_w = image.shape[0], image.shape[1]
        image = letterbox(image, new_shape=480)[0]
        im0s = deepcopy(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        image = image.float()
        image /= 255.0
        return image, im0s, img_h, img_w

    def frame_detect(self, frame):
        #   Image loading
        image, im0s, img_h, img_w = self.image_loading(frame)
        #   Yolo inference
        pred = self.yolo(image)
        rois = non_max_suppression(pred['rois'][0],self.cfg['nms']['conf_thres'], self.cfg['nms']['iou_thres'], classes= self.cfg['nms']['classes'],agnostic=self.cfg['nms']['agnostic_nms'])
        boxes = rois[0][:,:4]#  rois
        if len(boxes) == 0:
            return [], [], im0s
        feature_map = featuremapPack(pred['feature_map']) #   extract feature map and boxes
        
        #   Resize to bounding box
        h = int(boxes[0][2] - boxes[0][0])
        w = int(boxes[0][3] - boxes[0][1])
        resize_to_bbs = transforms.Compose([transforms.Resize((w, h))])

        #   Predict mask
        mask_logits = self.maskrcnn(feature_map, boxes)
        mask_logits = resize_to_bbs(mask_logits)
        mask_logits = mask_logits.detach().cpu().numpy()
        mask_logits = mask_logits[0][0]>0.5
        mask_logits.dtype = 'uint8'
        mask_logits = mask_logits*255
        mask_logits = cv2.cvtColor(mask_logits, cv2.COLOR_GRAY2RGB)

        return boxes, mask_logits, im0s



def vido_detect(video_path, save_path, maskrcnn_wts, cfg):
    print('Start to detect {}'.format(video_path))
    print(' ')
    Detector = VideoDetection(maskrcnn_wts, cfg)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_name = (video_path.split('/')[-1]).split('.')[0]
    out = cv2.VideoWriter(os.path.join(save_path, '{}.avi'.format(video_name)), fourcc, fps, (480,480))
    frame_id = 0
    while cap.isOpened():
        try:
            start = time.time()
            ret, frame = cap.read()
            boxes, mask_logits, im0s = Detector.frame_detect(frame)
            
            #   Draw mask in original image
            if len(boxes) != 0:
                mask_h, mask_w = mask_logits.shape[0], mask_logits.shape[1]
                cv2.rectangle(im0s, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)
                im0s_roi = im0s[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)]
                det = cv2.addWeighted(im0s_roi,0.7 ,mask_logits, 0.3, 0)
                im0s[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)] = det
                cv2.imwrite('{}/{}.jpg'.format(save_path, frame_id), im0s)
            out.write(im0s)
            
            '''
            #   Draw mask in mask
            #       Generate an empty mask in original size (480, 480)
            zeros =  np.zeros([im0s.shape[0], im0s.shape[1], 3],dtype='uint8')
            if len(boxes) != 0:
                mask_h, mask_w = mask_logits.shape[0], mask_logits.shape[1]
                zeros_roi = zeros[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)]                
                det = cv2.addWeighted(zeros_roi,0 ,mask_logits, 1, 0)
                zeros[int(boxes[0][1]):int(boxes[0][1]+mask_h), int(boxes[0][0]):int(boxes[0][0]+mask_w)] = det
                cv2.imwrite('{}/{}.jpg'.format(save_path, frame_id), zeros)
            out.write(zeros)
            '''
            end = time.time()
            print('FPS = {} ms'.format((end-start)*1000))
            frame_id += 1
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



if __name__ == '__main__':
    video_dir = 'Video'
    video_save = 'Detection'
    roimodelPath = 'run/0822_2/best_0822.pt'
    cfgPath = '/home/insign/Doc/insign/Mask_yolo/config/yolocfg.yaml'
    device = 'cuda'

    #   Loading cfg
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    #   read video 
    for video in os.listdir(video_dir):
        video_path  = os.path.join(video_dir, video)
        #   check save path
        if os.path.isfile(video_path):
            save = os.path.join(video_save, video.split('.')[0])
            if not os.path.isdir(save):
                os.mkdir(save)
            vido_detect(video_path, save, roimodelPath, cfg)