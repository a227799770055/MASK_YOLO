import torch
from torch import nn
import sys
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import TRTModule

class yoloModelPack(nn.Module):
    def __init__(self, backbone, fpn, pan, detector):
        super(yoloModelPack, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pan = pan
        self.detector = detector
    
    def forward(self, x):
        res = {}
        feat_map= []
        out = self.backbone(x)
        out = self.fpn(out)
        out1 = self.pan(out)
        out2 = self.detector(list(out1))
        
        feat_map.append(out1[0])
        feat_map.append(out1[1])
        feat_map.append(out1[2])
        res['feature_map'] = (feat_map)
        res['rois'] = out2
        return res

class yoloModelPack2TRT(nn.Module):
    def __init__(self, backbone, detector):
        super(yoloModelPack2TRT, self).__init__()
        self.backbone = backbone
        self.detector = detector
    
    def forward(self, x):
        res = {}
        feat_map= []
        out1 = self.backbone(x)
        out2 = self.detector(list(out1))
        
        feat_map.append(out1[0])
        feat_map.append(out1[1])
        feat_map.append(out1[2])
        res['feature_map'] = (feat_map)
        res['rois'] = out2
        return res
    
class maskModelPack2TRT(nn.Module):
    def __init__(self, featurealign, roipool, headdetector, image_shapes=[(640,640)]):
        super(maskModelPack2TRT, self).__init__()
        self.featurealign = featurealign
        self.roipool = roipool
        self.headdetector = headdetector
        self.image_shapes = image_shapes
    def forward(self, feat1, feat2, feat3, boxes):
        f1, f2, f3 = self.featurealign(feat1, feat2, feat3)
        features = {}
        key_name = ["feat1","feat2","feat3"]
        feature_map = [f1,f2,f3]
        for i,j in zip(key_name, feature_map):
                features[i] = j
        featpool = self.roipool(features, [boxes], self.image_shapes)
        mask = self.headdetector(featpool)
        return mask
        
def loadTRTmodel(path):
    model = TRTModule()
    model.load_state_dict(torch.load(path))
    return model 