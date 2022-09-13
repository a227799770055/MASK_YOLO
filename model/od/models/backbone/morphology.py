import torch
import torch.nn as nn
import sys
sys.path.append('/home/insign/Doc/insign/flexible-yolov5/od')
from od.models.modules.common import Focus, Conv, C3, SPP, BottleneckCSP, C3TR, MorphologyCNN
from utils.general import make_divisible


class Morphology(nn.Module):
    def __init__(self, focus=False, version='L', with_C3TR=False):
        super(Morphology, self).__init__()
        self.version = version
        self.with_focus = focus
        self.with_c3tr = with_C3TR
        gains = {'s': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'l': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1.33, 'gw': 1.25}}
        self.gd = gains[self.version.lower()]['gd']  # depth gain
        self.gw = gains[self.version.lower()]['gw']  # width gain

        # self.y_channel_out = {
        #     'stage1': 64,
        #     'stage2_1': 128,
        #     'stage2_2': 128,
        #     'stage3_1': 256,
        #     'stage3_2': 256,
        #     'stage4_1': 512,
        #     'stage4_2': 512,
        #     'stage5': 1024,
        #     'spp': 1024,
        #     'csp1': 1024,
        #     'conv1': 512
        # }
        

        self.m_channels_out = {
            'm2':16,
            'm3':32,
            'm4':64,
            'm5':128
        }

        self.y_channel_out = {
            'stage1': 32,
            'stage2_1': 64,
            'stage2_2': 64,
            'stage3_1': 128,
            'stage3_2': 128,
            'stage4_1': 256,
            'stage4_2': 256,
            'stage5': 512,
            'spp': 512,
            'csp1': 512,
            'conv1': 256
        }
        self.channels_out = {
            'stage1': 32,
            'stage2_1': 64,
            'stage2_2': 64,
            'stage3_1': 128,
            'stage3_2': self.y_channel_out['stage3_2'] + self.m_channels_out['m3'],
            'stage4_1': 256,
            'stage4_2': self.y_channel_out['stage4_2'] + self.m_channels_out['m4'],
            'stage5': 512,
            'spp': 512,
            'csp1': self.y_channel_out['csp1'] + self.m_channels_out['m5'],
            'conv1': 256
        }
        self.re_channels_out()

        if self.with_focus:
            self.stage1 = Focus(3, self.channels_out['stage1'])
        else:
            self.stage1 = Conv(3, self.channels_out['stage1'], 3, 2)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(self.channels_out['stage1'], self.channels_out['stage2_1'], k=3, s=2)
        self.stage2_2 = C3(self.channels_out['stage2_1'], self.channels_out['stage2_2'], self.get_depth(3))
        self.stage3_1 = Conv(self.channels_out['stage2_2'], self.channels_out['stage3_1'], 3, 2)
        self.stage3_2 = C3(self.channels_out['stage3_1'], self.y_channel_out['stage3_2'], self.get_depth(9))
        self.stage4_1 = Conv(self.y_channel_out['stage3_2'], self.channels_out['stage4_1'], 3, 2)
        self.stage4_2 = C3(self.channels_out['stage4_1'],self.y_channel_out['stage4_2'], self.get_depth(9))
        self.stage5 = Conv(self.y_channel_out['stage4_2'], self.channels_out['stage5'], 3, 2)
        self.spp = SPP(self.channels_out['stage5'], self.channels_out['spp'], [5, 9, 13])
        if self.with_c3tr:
            self.c3tr = C3TR(self.channels_out['spp'], self.y_channel_out['csp1'], self.get_depth(3), False)
        else:
            self.csp1 = C3(self.channels_out['spp'], self.y_channel_out['csp1'], self.get_depth(3), False)
        

        #Morphology branch
        
        self.m2 = MorphologyCNN(3,self.m_channels_out['m2'],3,4)
        self.m2_1 = MorphologyCNN(self.m_channels_out['m2'],self.m_channels_out['m2'],3,1)
        self.m3 = MorphologyCNN(self.m_channels_out['m2'],self.m_channels_out['m3'],3,2)
        self.m4 = MorphologyCNN(self.m_channels_out['m3'],self.m_channels_out['m4'],3,2)
        self.m5 = MorphologyCNN(self.m_channels_out['m4'],self.m_channels_out['m5'],3,2)


        self.conv1 = Conv((self.channels_out['csp1']), self.channels_out['conv1'], 1, 1)
        self.out_shape = {'C3_size': self.channels_out['stage3_2'],
                          'C4_size': self.channels_out['stage4_2'],
                          'C5_size': self.channels_out['conv1']}
        print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['stage3_2'],
                                                                    self.channels_out['stage4_2'],
                                                                    self.channels_out['conv1']))

    def forward(self, x):
        #Morphology
        mx = self.m2(x)
        mx = self.m2_1(mx)
        mc3 = self.m3(mx)
        mc4 = self.m4(mc3)
        mc5 = self.m5(mc4)

        #Darknet53
        x = self.stage1(x)
        x21 = self.stage2_1(x)
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22)
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3)
        c4 = self.stage4_2(x41)
        x5 = self.stage5(c4)
        spp = self.spp(x5)
        if not self.with_c3tr:
            csp1 = self.csp1(spp)
        else:
            c3tr = self.c3tr(spp)
        

        #concate
        c3o = torch.cat((mc3, c3), dim=1)
        c4o = torch.cat((mc4, c4),dim=1)
        c5o = torch.cat((mc5, csp1), dim=1)
        c5o = self.conv1(c5o)
        return c3o, c4o, c5o

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)


if __name__ == '__main__':
    model = Morphology()

    t = torch.ones(1,3,640,640)
    y1, y2, y3 = model(t)
    print(y1.shape, y2.shape, y3.shape)