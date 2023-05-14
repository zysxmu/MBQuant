import torch
import torch.nn as nn
import logging
import os
from torch.hub import load_state_dict_from_url
from model_cifar.quant_conv import QConv, QLinear, QConv3x3, QConv1x1, SwitchableBatchNorm2d
from .shuffle_utils import channel_shuffle, CyclicShuffle

__all__ = ['resnet20_quant', 'resnet8_quant']

class QBasicBlock4Cifar(nn.Module):
    def __init__(self, args, inplanes, planes, stride=1, groups=1, dilation=1,
                 norm_layer=None, oneBit_outchannel=-1, oneBit_inchannel=-1,
                 last_conv=False, first_conv=False, r8=False):
        super(QBasicBlock4Cifar, self).__init__()

        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
        self.oneBit_outchannel = oneBit_outchannel
        self.oneBit_inchannel = oneBit_inchannel
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.inplanes = inplanes
        self.r8 = r8
        self.planes = planes
        assert not self.last_conv or not self.first_conv

        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if first_conv:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                                  first_conv=first_conv)
        else:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel)

        self.bn1 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel)
        self.relu = nn.ReLU(inplace=True)

        if last_conv:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel,
                                  last_conv=last_conv)
            self.bn2 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel, last_conv=last_conv)

        else:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel)
            self.bn2 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if r8:
                self.shortcut = nn.Sequential(
                QConv1x1(args, inplanes, planes, stride, groups=self.groups,
                         oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel, last_conv=last_conv),
                norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel, last_conv=last_conv),
            )
            else:
                self.shortcut = nn.Sequential(
                        QConv1x1(args, inplanes, planes, stride, groups=self.groups,
                            oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel),
                        norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel),
                        )

    def select(self, identity):

        if (self.weight_bit & 1) == 0:
            index = self.inplanes // self.groups * (self.weight_bit // 2)
            return identity[:, :index, :, :]
        else:
            index = self.inplanes // self.groups * (self.weight_bit // 2) + self.oneBit_outchannel
            return identity[:, :index, :, :]

    def addZeroS(self, identity):
        if (self.weight_bit & 1) == 0:
            return identity
        else:
            if self.r8:
                index = self.planes // self.groups - self.oneBit_outchannel
            else:
                index = self.inplanes // self.groups - self.oneBit_outchannel
            zeros = torch.zeros(identity.shape[0], index, identity.shape[2], identity.shape[3]).cuda()
            out = torch.cat((identity, zeros), dim=1)
            return out

    def forward(self, x):
        identity = x

        if self.first_conv:
            identity = self.select(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.last_conv:
            identity = self.addZeroS(identity)
        # out = channel_shuffle(out, self.weight_bit // 2, self.weight_bit)
        out += self.shortcut(identity)
        out = self.relu(out)

        if self.last_conv:
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)
            out = torch.mean(torch.stack(out, dim=0), dim=0)

        return out


class QResNet4Cifar(nn.Module):
    def __init__(self, args, block, layers, num_classes=10, groups=1, norm_layer=None):
        super(QResNet4Cifar, self).__init__()

        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.inplanes = 16 * self.groups

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if layers == [1, 1, 1]: # for resnet8
            self.layer1 = self._make_layer(args, block, 16 * self.groups, layers[0], stride=1,
                                           oneBit_outchannel=16, oneBit_inchannel=8, first_conv=True)
            self.layer2 = self._make_layer(args, block, 32 * self.groups, layers[1], stride=2,
                                           oneBit_outchannel=24, oneBit_inchannel=16)
            self.layer3 = self._make_layer(args, block, 64 * self.groups, layers[2], stride=2,
                                           oneBit_outchannel=48, oneBit_inchannel=24, last_conv=True)
        else: # for resnet20
            self.layer1 = self._make_layer(args, block, 16 * self.groups, layers[0], stride=1,
                                           oneBit_outchannel=16, oneBit_inchannel=8, first_conv=True)
            self.layer2 = self._make_layer(args, block, 32 * self.groups, layers[1], stride=2,
                                           oneBit_outchannel=24, oneBit_inchannel=16)
            self.layer3 = self._make_layer(args, block, 64 * self.groups, layers[2], stride=2,
                                           oneBit_outchannel=48, oneBit_inchannel=24, last_conv=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, args, block, planes, blocks, stride=1,
                    oneBit_outchannel=-1, oneBit_inchannel=-1, last_conv=False, first_conv=False):
        norm_layer = self._norm_layer
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        # for resnet-8
        if len(strides) == 1:
            layers.append(block(args, self.inplanes, planes, stride, self.groups,
                                norm_layer=norm_layer,
                                oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                                first_conv=first_conv, last_conv=last_conv, r8=True
                                ))
            self.inplanes = planes
        else:
            for i, stride in enumerate(strides):
                if i == 0:
                    layers.append(block(args, self.inplanes, planes, stride, self.groups,
                                        norm_layer=norm_layer,
                                        oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                                        first_conv=first_conv
                                        ))
                elif i == len(strides)-1:
                    layers.append(block(args, self.inplanes, planes, stride, self.groups,
                                        norm_layer=norm_layer,
                                        oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                                        last_conv=last_conv
                                        ))
                else:
                    layers.append(block(args, self.inplanes, planes, stride, self.groups,
                                        norm_layer=norm_layer,
                                        oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel
                                        ))

                self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.tile(1, self.groups, 1, 1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet20_quant(args, **kwargs):
    return QResNet4Cifar(args, QBasicBlock4Cifar, [3, 3, 3], **kwargs)


def resnet8_quant(args, **kwargs):
    return QResNet4Cifar(args, QBasicBlock4Cifar, [1, 1, 1], **kwargs)

