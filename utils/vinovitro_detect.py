import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import os, sys
import numpy as np
from PIL import Image
import traceback


def detector(inputs, model, processes):
    class_names = ["vivo", "vitro"]
    soft = torch.nn.Softmax(dim=1)
    
    inputs = image_process(inputs, processes)

    outputs = model(inputs)

    out = soft(outputs)
    out = out.cpu().detach().numpy()
    out = out[0]>0.9
    pred = np.where(out==True)[0]

    if len(pred) != 0:
        label = class_names[pred[0]]
    else:
        label = 'blank'

    return label


def image_process(image, processes):
    frame_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_copy = Image.fromarray(frame_copy)
    inputs = processes(frame_copy).unsqueeze(0).to("cuda")
    return inputs


if __name__ == "__main__":
    video_path = "/media/insign/Transcend/Workspace/Dataset/1_polyp_dataset/收集的資料庫/秀傳/raw videos/秀傳影片_2207/OK"
    model_path = "/home/insign/Doc/insign/Mask_yolo/data/vitrovivo_classified_resnet101_1201.pth"
    save_path = "media/insign/Transcend/Workspace/Dataset/3_vitrovivo/test_video"

    #   load model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path,map_location=device)
    model = model.eval()

    #   frame process
    preprocess=transforms.Compose([
                transforms.Resize(size=512),
                transforms.CenterCrop(size=512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # 
    videos = os.listdir(video_path)
    for i in videos:
        video = os.path.join(video_path,i)

        #   loading video
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_name = (video.split('/')[-1]).split('.')[0]
        out = cv2.VideoWriter(os.path.join(save_path, '{}.mp4'.format(video_name)), fourcc, fps, (width, height))

        threshold = {'vivo': False, 'vitro':False, 'blank':False}
        frame_count = {'vivo': 0, 'vitro':0, 'blank':0}
        frame_label = {'vivo': 0, 'vitro':0, 'blank':0}
        frameID = 0
        precent_locat = 'blank'

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                label = detector(frame, model, preprocess)

                # 計算label連續出現的幀數，當沒有連貫時，幀數歸0
                # frame_label 紀錄上一幀的位置
                # frame_count 紀錄連續出現幾幀
                if frame_label[label]==0:
                    frame_label[label]=frameID
                elif frame_label[label] == frameID-1:
                    frame_label[label]=frameID
                    frame_count[label] = frame_count[label]+1
                elif frame_label[label] != frameID-1:
                    frame_count[label] = 0
                    frame_label[label] = 0
                
                # 當特定label的幀數連續出現且達到閥值時，該label的閥門就會變成true
                if threshold[label]==False and frame_count[label]>10: # threshold setting as 60 frames
                    threshold[label]=True
                
                if threshold[label]==True:
                    cv2.putText(frame, str(label), (100, 100), cv2.FONT_ITALIC, 
                                    1, (255, 255, 255), 2, cv2.LINE_AA)


            # 寫入影片
                out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                frameID += 1
            
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
