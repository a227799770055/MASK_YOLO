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
