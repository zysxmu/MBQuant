'''
from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
'''
import math

import torch.nn as nn
import torch.nn.init as init
from model_cifar.quant_conv import QConv, SwitchableBatchNorm2d
import torch

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
__all__ = [
    'vgg11_bn_quant'
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # import IPython
        # IPython.embed()

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Tile(nn.Module):
    def __init__(self, args):
        super(Tile, self).__init__()
        self.groups = args.groups

    def forward(self, x):
        x = x.tile(1, self.groups, 1, 1)
        return x

class LastSum(nn.Module):
    def __init__(self, args):
        super(LastSum, self).__init__()
        self.groups = args.groups
        self.weight_bit = 2

    def forward(self, x):
        x = x.chunk((self.weight_bit + 1) // 2, dim=1)
        x = torch.mean(torch.stack(x, dim=0), dim=0)
        return x


def make_layers(args, cfg, batch_norm=False):
    layers = []
    in_channels = 3
    oneBit =  ['xxx', 'M', 107, 'M', 215, 215, 'M', 224, 224, 'M', 224, 512, 'M']
    for idx, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if idx == 0: # first conv retain as fp
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [Tile(args=args)]
            else:
                if oneBit[idx - 1] == 'M':
                    idx_pre = idx - 2
                else:
                    idx_pre = idx - 1
                v = v * args.groups
                if idx == 2: # first quantized conv
                    conv2d = QConv(in_channels, v, 3, args=args, padding=1, bias=False, groups=args.groups,
                                   oneBit_outchannel=oneBit[idx], oneBit_inchannel=oneBit[idx_pre], first_conv=True)
                    if batch_norm:
                        layers += [conv2d,
                                   SwitchableBatchNorm2d(v, groups=args.groups, oneBit_outchannel=oneBit[idx]),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]

                elif idx == len(cfg)-2: # last conv
                    conv2d = QConv(in_channels, v, 3, args=args, padding=1, bias=False, groups=args.groups,
                                   oneBit_outchannel=oneBit[idx], oneBit_inchannel=oneBit[idx_pre], last_conv=True)
                    if batch_norm:
                        layers += [conv2d,
                                   SwitchableBatchNorm2d(v, groups=args.groups, oneBit_outchannel=oneBit[idx],
                                                         last_conv=True),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                else:
                    conv2d = QConv(in_channels, v, 3, args=args, padding=1, bias=False, groups=args.groups,
                                   oneBit_outchannel=oneBit[idx], oneBit_inchannel=oneBit[idx_pre])
                    if batch_norm:
                        layers += [conv2d,
                                   SwitchableBatchNorm2d(v, groups=args.groups, oneBit_outchannel=oneBit[idx]),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v
    layers += [LastSum(args=args)]
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

def vgg11_bn_quant(args, num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(args, cfg['A'], batch_norm=True), num_classes)
