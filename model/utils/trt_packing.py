from torch2trt import TRTModule
from torch import nn


class packingModel(nn.Module):
    def __init__(self, backbone, head):
        super(packingModel, self).__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        out = self.backbone(x)
        out = list(out)
        out = self.head(out)
        return out
