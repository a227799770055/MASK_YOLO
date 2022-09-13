from torch import nn

class ModelPack(nn.Module):
    def __init__(self, backbone, fpn, pan, detector):
        super(ModelPack, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pan = pan
        self.detector = detector
        # self.featurealign1 = nn.Conv2d(256, 1024, 1, 1).to('cuda')
        # self.featurealign2 = nn.Conv2d(512, 1024, 1, 1).to('cuda')
    
    def forward(self, x):
        res = {}
        feat_map= []
        out = self.backbone(x)
        out = self.fpn(out)
        out1 = self.pan(out)
        out2 = self.detector(list(out1))
        # feat_map.append(self.featurealign1(out1[0]))
        # feat_map.append(self.featurealign2(out1[1]))
        feat_map.append(out1[0])
        feat_map.append(out1[1])
        feat_map.append(out1[2])
        res['feature_map'] = (feat_map)
        res['rois'] = out2
        return res