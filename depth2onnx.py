import torch
import os, sys
sys.path.append('/home/insign/Doc/insign/Mask_yolo')
from typing import Any
import argparse
import mmcv
from mmcv.runner import load_checkpoint
from depth.models import build_depther
from torchvision import transforms
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import onnxruntime as rt
import numpy as np
import onnx
from onnxsim import simplify

def load_depth_model(cfg, pth_path):
    model = build_depther(
        cfg.model,
        test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, pth_path)
    return model

def depth_estimation(image, model):
    # model = MMDataParallel(model, device_ids=[0])
    ori_h, ori_w =  image.shape[2], image.shape[3]
    model.eval()
    resize_to_512 = transforms.Compose([transforms.Resize((512, 512))])
    with torch.no_grad():
        input_img = resize_to_512(image)
        
        result = model(img = [input_img])[0][0]
        result = cv2.resize(result, dsize=(ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    return result

def load_depth_cfg(path):
    cfg = mmcv.Config.fromfile(path)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    return cfg

def getFakeMeta(img_shape = (512, 512, 3)):
        meta = dict()
        meta['filename'] = 'test.png'
        meta['ori_filename'] = 'test.png'
        meta['ori_shape'] = img_shape
        meta['img_shape'] = img_shape
        meta['pad_shape'] = img_shape
        meta['scale_factor'] = [1., 1., 1., 1.]
        meta['flip'] = False
        meta['flip_direction'] = 'horizontal'
        meta['to_rgb'] = True
        return [meta]

class Head(torch.nn.Module):
    def __init__(self, decode_head):
        super(Head, self).__init__()
        self.metadata = getFakeMeta()
        self.head_model = decode_head
    def forward(self, x):
        return self.head_model(x, self.metadata)

if __name__ == '__main__':
    # Loading model
    depth_cfg=load_depth_cfg('./data/depthformer_swint_w7_endoscopy.py')
    depth_cfg2='/home/insign/Doc/insign/Mask_yolo/data/depth_test_1115.pth'
    depthformer_net = load_depth_model(depth_cfg, depth_cfg2).eval()

    input_var = torch.rand(1, 3, 512, 512)
    img_meta = getFakeMeta()
    backbone = depthformer_net.backbone
    neck = depthformer_net.neck
    decode_head = depthformer_net.decode_head

    # y1 = backbone(input_var)
    # y1=[torch.rand(1, 64, 256, 256), torch.rand(1,96,128,128), torch.rand(1,192,64,64), torch.rand(1,384,32,32), torch.rand(1,768,16,16)]
    # head = Head(decode_head)
    

    toONNX = False
    if toONNX:
        # torch.onnx.export(depthformer_net, input_var,
        #             "test.onnx"	,
        #             opset_version=14,
        #             do_constant_folding=True,	# 是否执行常量折叠优化
        #             input_names=["input"],	# 输入名
        #             output_names=["output"],	# 输出名
        # )

        y1=[torch.rand(1, 64, 256, 256), torch.rand(1,96,128,128), torch.rand(1,192,64,64), torch.rand(1,384,32,32), torch.rand(1,768,16,16)]
        torch.onnx.export(neck, y1,
                    "test_neck.onnx"	,
                    opset_version=14,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
        )

        # y2=[torch.rand(1, 64, 256, 256), torch.rand(1,96,128,128), torch.rand(1,192,64,64), torch.rand(1,384,32,32), torch.rand(1,768,16,16)]
        # torch.onnx.export(head, y2,
        #             "test_head.onnx"	,
        #             opset_version=14,
        #             do_constant_folding=True,	# 是否执行常量折叠优化
        #             input_names=["input"],	# 输入名
        #             output_names=["output"],	# 输出名
        # )
