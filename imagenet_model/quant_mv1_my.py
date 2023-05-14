import torch
import torch.nn as nn
import math
import numpy as np
from torch.hub import load_state_dict_from_url
from imagenet_model.quant_conv import QConv, QConv_Tra_Mulit, SwitchableBatchNorm2d, SwitchableBatchNorm2d_Tra_Mulit
from .shuffle_utils import channel_shuffle_mv1

__all__ = ['mv1_quant']


class QDepthwiseSeparableConv(nn.Module):
    def __init__(self, args, inp=0, outp=0, stride=0,
                 oneBit_outchannel=0, oneBit_inchannel=0,
                 last_conv=False, first_conv=False, channel_shuffle=False):
        super(QDepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.weight_bit = 2
        self.act_bit = 8
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.basewidth = inp // args.groups
        self.channel_shuffle = channel_shuffle

        layers = [
            QConv_Tra_Mulit(inp, inp, kernel_size=3, stride=stride,
                            padding=1, groups=inp, bias=False, dilation=1, args=args,
                            oneBit_outchannel=oneBit_inchannel,
                            basewidth=self.basewidth,
                            first_conv=first_conv),
            SwitchableBatchNorm2d_Tra_Mulit(inp, basewidth=self.basewidth, oneBit_outchannel=oneBit_inchannel),
            nn.ReLU(inplace=True),

            QConv(inp, outp, kernel_size=1, stride=1,
                     padding=0, groups=args.groups, bias=False, args=args,
                     oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                     last_conv=last_conv),
            SwitchableBatchNorm2d(outp, groups=args.groups, oneBit_outchannel=oneBit_outchannel),
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        # if self.channel_shuffle:
        #     out = channel_shuffle_mv1(out, self.weight_bit // 2, self.weight_bit)
        if self.last_conv:
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)
            out = torch.mean(torch.stack(out, dim=0), dim=0)
        return out


class Model(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        oneBit = [52, 40, 101, 148, 223, 373, 404, 236, 376, 419, 233, 892, 1024]
        # oneBit = [56, 43, 108, 156, 240, 404, 435, 250, 404, 450, 250, 962, 1024]
        j = 0
        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
            nn.Conv2d(
                3, channels, kernel_size=3,
                stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.groups = args.groups
        channels = channels * self.groups
        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c * self.groups
            if idx == len(self.block_setting) - 1:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1, last_conv=True,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    channels = outp
                    j += 1
            elif idx == 0:
                for i in range(n):
                    assert n == 1
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    oneBit_inchannel=32, oneBit_outchannel=oneBit[j], first_conv=True))
                    channels = outp
                    j += 1
            else:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    channels = outp
                    j += 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Linear(1024, num_classes)  # the last layer uses fp weights

        self.reset_parameters()
        self.i = 0
        self.args = args

    def forward(self, x):
        # if self.i % 50 == 0:
        #     print(self.i, x.shape)
        # self.i += 1
        # print(x[-1][0][0])
        x = self.head(x)
        # print(self.head[1].running_mean, self.head[1].running_var,
        #       self.head[1].track_running_stats, self.head[1].weight, self.head[1].bias)
        # import IPython
        # IPython.embed()
        # print('1', x[-1][0][0])
        # import sys
        # sys.exit()
        x = x.tile(1, self.groups, 1, 1)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def reset_parameters(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mv1_quant(args, **kwargs):
    return Model(args)
