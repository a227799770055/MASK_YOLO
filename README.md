# MASK_YOLO

## Dataset Preprocessing
---
如果要修改 training dataset 路徑，請修改 /config/config.yaml
> root:Polyp
> img_dir:Images
> mask_dir:Masks

 ./Polyp/Images
 ./Polyp/Masks

## Train
---
```bash
python3 utils/train.py
```

## Inference
---
```bash
python3 utils/inference.py
```
output path ./result

## Video Detection
```bash
python3 utils/detect_video_sort.py  \
        --cfgPath config/file/path \
        --videoPath video/folder/path \
        --savePath result/save/path
```

## Pytorch model convert to TensorRT

### 將 yolo 權重轉換成 tensorRT
#### pytorch weights 下載：[link](https://83516952-my.sharepoint.com/:f:/g/personal/eddie_hsiao_insign-medical_com/ElBW98W59IRFn0qhxtvYL0YBx9fSzvSwTKMFjFyc2p1e6g?e=EgAuHj)
```bash
python3 toTRT/yolo2trt.py --yoloPath yolo/weight/path
```
會儲存兩個權重檔
- toTRT/MorphYolo_backbone.pth
- toTRT/MorphYolo_head.pth

### 將 maskRCNN 權重轉換成 tensorRT
```bash
python3 toTRT/maskrcnn.py --maskPath maskrcnn/weight/path
```
會儲存三個權重檔
- toTRT/featurealign.pth
- toTRT/roipool.pth
- toTRT/headdetector.pth

[validation data](https://83516952-my.sharepoint.com/:f:/g/personal/eddie_hsiao_insign-medical_com/EnkRfcBYwc5Esu60mPMG3KsBZ7mz5NZ9x9-igITwBQRFjw?e=dqYz93)

