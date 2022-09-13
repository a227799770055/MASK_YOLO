import torch
from torch import nn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads 
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch.nn.functional as F
import warnings
from typing import Tuple, List, Dict, Optional, Union

''' 
m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
i = collections.OrderedDict()
i['feat1'] = torch.rand(1, 5, 64, 64)
i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
i['feat3'] = torch.rand(1, 5, 16, 16)

boxes = torch.rand(6, 4) * 256
boxes[:, 2:] += boxes[:, :2]
image_sizes = [(512, 512)]
output = m(i, [boxes], image_sizes)
print(output.shape)

'''
# TODO
class Mask_head(nn.Module):

    def __init__(self,
                num_classes=1,
                out_channels = 512,
                # Box parameters 
                box_roi_pool=None,
                box_head=None,
                box_predictor=None,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,
                box_detections_per_img=100,
                box_fg_iou_thresh=0.5,
                box_bg_iou_thresh=0.5,
                box_batch_size_per_image=512,
                box_positive_fraction=0.25,
                bbox_reg_weights=None,
                # Mask parameters
                mask_roi_pool=None,
                mask_head=None,
                mask_predictor=None):
        super().__init__()

        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
        
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)
        
        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections
    
    def forward(self, features, proposals, image_sizes, targets, original_image_sizes=None):
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        # detections = self.transform.postprocess(detections, image_sizes, original_image_sizes)
        
        losses = {}
        losses.update(detector_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
    
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

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



if __name__ == '__main__':
    model = Mask_head()
    params = [p for p in model.parameters() if p.requires_grad]
    print(model)
