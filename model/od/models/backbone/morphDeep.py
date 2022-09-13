import torch
import torch.nn as nn
from od.models.modules.common import Focus, Conv, C3, SPP, BottleneckCSP, C3TR, MorphologyCNN
from utils.general import make_divisible


class morphDeep(nn.Module):
    def __init__(self, focus=False, version='L', with_C3TR=False):
        super(morphDeep, self).__init__()
        self.version = version
        self.with_focus = focus
        self.with_c3tr = with_C3TR
        gains = {'s': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'l': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1.33, 'gw': 1.25}}
        self.gd = gains[self.version.lower()]['gd']  # depth gain
        self.gw = gains[self.version.lower()]['gw']  # width gain

        self.channels_out = {
            'stage1': 64,
            'stage2_1': 128,
            'stage2_2': 128,
            'stage3_1': 256,
            'stage3_2': 256,
            'stage4_1': 512,
            'stage4_2': 512,
            'stage5': 1024,
            'spp': 1024,
            'csp1': 1024,
            'conv1': 512
        }
        self.m_channels_out = {
            'm2':16,
            'm3':32,
            'm4':64,
            'm5':128
        }
        self.re_channels_out()

        


        # for latest yolov5, you can change BottleneckCSP to C3
  
        

        #Morphology branch
        
        self.m2 = MorphologyCNN(3,self.m_channels_out['m2'],3,4)
        self.m2_1 = MorphologyCNN(self.m_channels_out['m2'],self.m_channels_out['m2'],3,1)
        self.m3 = MorphologyCNN(self.m_channels_out['m2'],self.m_channels_out['m3'],3,2)
        self.m4 = MorphologyCNN(self.m_channels_out['m3'],self.m_channels_out['m4'],3,2)
        self.m5 = MorphologyCNN(self.m_channels_out['m4'],self.m_channels_out['m5'],3,2)

        self.out_shape = {'C3_size': self.m_channels_out['m3'],
                          'C4_size': self.m_channels_out['m4'],
                          'C5_size': self.m_channels_out['m5']}


        print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.m_channels_out['m3'],
                                                                    self.m_channels_out['m4'],
                                                                    self.m_channels_out['m5']))


    def forward(self, x):
        #Morphology
        mx = self.m2(x)
        mx = self.m2_1(mx)
        mc3 = self.m3(mx)
        mc4 = self.m4(mc3)
        mc5 = self.m5(mc4)

        return mc3, mc4, mc5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.m_channels_out.items():
            self.m_channels_out[k] = self.get_width(v)


if __name__ == '__main__':
    model = Morphology()

    t = torch.ones(1,3,640,640)
    y1, y2, y3 = model(t)
    print(y1.shape, y2.shape, y3.shape)