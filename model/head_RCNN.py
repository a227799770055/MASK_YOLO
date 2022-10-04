import torch
from torch import nn 
import os
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import json

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)

class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d[f"mask_fcn{layer_idx}"] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation
            )
            d[f"relu{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super().__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)



class ROIHeadsMask(nn.Module):
    def __init__(
        self,
        num_classes=1,
        out_channels=512,
        mask_roi_pool=None,
        mask_head=None, 
        mask_predictor=None,
        fcn_predictor = None
         ):
        super().__init__()

        self.featurealign1 = nn.Conv2d(256, 512, 1, 1).to('cuda')
        self.featurealign2 = nn.Conv2d(512, 512, 1, 1).to('cuda')
        self.featurealign3 = nn.Conv2d(1024, 512, 1, 1).to('cuda')

        if mask_roi_pool is None:
            self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["feat1","feat2","feat3"] ,output_size=28, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            self.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)    

        #   使用 FCN Head 取代 mask predictor    
        if fcn_predictor is None:
            fcn_predictor_in_channel = 256
            self.fcn_predictor = FCNHead(fcn_predictor_in_channel, num_classes)
    
    def forward(
        self,
        features_0,
        features_1,
        features_2,
        proposals,
        image_shapes=[(480,480)],
        targets=None):

        mask_proposals = [proposals]
    
        features_0 = self.featurealign1(features_0)
        features_1 = self.featurealign2(features_1)
        features_2 = self.featurealign3(features_2)
        
        features = {}
        key_name = ["feat1","feat2","feat3"]
        feature_map = [features_0,features_1,features_2]
        for i,j in zip(key_name, feature_map):
            features[i] = j
        
        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            # mask_logits = self.mask_predictor(mask_features)
            mask_logits = self.fcn_predictor(mask_features)
        else:
            raise Exception("Except mask_roi_pool to be not None")
        
        return mask_logits

        # loss_mask = {}
        # if self.training:
        #     assert targets is not None
            

def convertPredList2Tensor(pred):    
    assert type(pred) == dict
    keys = pred.keys()
    for key in keys:
        lenList = len(pred[key])
        for i in range(lenList):
            pred[key][i] = torch.FloatTensor( pred[key][i])

# def featuremapPack(feature_map):
#     from collections import OrderedDict
#     map = OrderedDict()
#     key_name = ["feat1","feat2","feat3"]
#     for i,j in zip(key_name, feature_map):
#         map[i] = j
#     return map

def featuremapPack(feature_map):
    map = {}
    key_name = ["feat1","feat2","feat3"]
    for i,j in zip(key_name, feature_map):
        map[i] = j
    return map

if __name__ == '__main__':
    
    feat1 = torch.randn(1,512,60,60)
    feat2 = torch.randn(1,512,30,30)
    feat3 = torch.randn(1,512,15,15)
    feat = {
        'feat1':feat1,
        'feat2':feat2,
        'feat3':feat3
    }
    rois = torch.rand(1,4)
    #   convert list to tensor
    convertPredList2Tensor(feat)
    #   packing feature map
    feature_map = featuremapPack(feat)
    boxes = rois

    roiHead = ROIHeadsMask()
    y = roiHead(feature_map, boxes)
