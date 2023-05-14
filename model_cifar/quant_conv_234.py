import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv', 'QConv1x1', 'QConv3x3', 'SwitchableBatchNorm2d', 'QLinear']


class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out

    @staticmethod
    def backward(ctx, g):
        return g, None


# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, oneBit_outchannel=-1, oneBit_inchannel=-1, last_conv=False, first_conv=False):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.STE_discretizer = STE_discretizer.apply
        self.weight_bit = 2  # defalut
        self.act_bit = 2  # defalut
        self.oneBit_outchannel = oneBit_outchannel
        self.oneBit_inchannel = oneBit_inchannel
        self.last_conv = last_conv
        self.first_conv = first_conv

        if self.quan_weight:
            self.weight_levels = 2 ** 2
            self.uW = nn.ParameterList([nn.Parameter(data=torch.tensor(0).float()) for _ in range(4)])
            self.lW = nn.ParameterList([nn.Parameter(data=torch.tensor(0).float()) for _ in range(4)])
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())

        if self.quan_act:
            self.act_levels = -1
            self.uA = nn.ParameterList([nn.Parameter(data=torch.tensor([0, 0, 0, 0, 0, 0, 0]).float()) for _ in range(4)])
            self.lA = nn.ParameterList([nn.Parameter(data=torch.tensor([0, 0, 0, 0, 0, 0, 0]).float()) for _ in range(4)])
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.ParameterList([nn.Parameter(data=torch.tensor(1).float()) for _ in range(7)])
        # self.output_scale = nn.Parameter(data=torch.tensor(1).float())
        self.hook_Qvalues = False
        self.buff_weight = None
        self.buff_act = None
        assert not self.last_conv or not self.first_conv

    def weight_quantization(self, weight, group_index):
        if not self.quan_weight or self.weight_bit == 32:
            print('return weight, FP!')
            return weight
        weight = (weight - self.lW[group_index]) / (self.uW[group_index] - self.lW[group_index])
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels)
        weight = (weight - 0.5) * 2  # [-1, 1]
        return weight

    def act_quantization(self, x, group_index):
        if not self.quan_act or self.act_bit == 32:
            print('return x, FP!')
            return x

        index = self.act_bit - 2
        self.act_levels = 2 ** self.act_bit
        # self.act_levels = 2 ** 8
        x = (x - self.lA[group_index][index]) / (self.uA[group_index][index] - self.lA[group_index][index])
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def group_activation_quantization(self, FPact_1, FPact_2=None):

        if (self.weight_bit & 1) == 0:
            tmp_list = []
            FPact_1 = FPact_1.chunk(self.weight_bit // 2, dim=1)
            for i in range(len(FPact_1)):
                tmp = self.act_quantization(FPact_1[i], i)
                tmp_list.append(tmp)
            Qact_1 = torch.cat(tmp_list, dim=1)
            return Qact_1

        else:
            tmp_list = []
            FPact_1 = FPact_1.chunk(self.weight_bit // 2, dim=1)
            for i in range(len(FPact_1)):
                tmp = self.act_quantization(FPact_1[i], i)
                tmp_list.append(tmp)
            Qact_1 = torch.cat(tmp_list, dim=1)
            Qact_2 = self.act_quantization(FPact_2, self.weight_bit//2)
            return Qact_1, Qact_2

    def group_weight_quantization(self, FPweight_1, FPweight_2=None):

        if (self.weight_bit & 1) == 0:
            tmp_list = []
            FPweight_1 = FPweight_1.chunk(self.weight_bit // 2, dim=0)
            for i in range(len(FPweight_1)):
                tmp = self.weight_quantization(FPweight_1[i], i)
                tmp_list.append(tmp)
            Qweight_1 = torch.cat(tmp_list, dim=0)
            return Qweight_1

        else:
            tmp_list = []
            FPweight_1 = FPweight_1.chunk(self.weight_bit // 2, dim=0)
            for i in range(len(FPweight_1)):
                tmp = self.weight_quantization(FPweight_1[i], i)
                tmp_list.append(tmp)
            Qweight_1 = torch.cat(tmp_list, dim=0)
            Qweight_2 = self.weight_quantization(FPweight_2, self.weight_bit // 2)
            return Qweight_1, Qweight_2

    def initialize(self, x):

        FPweight, FPact = self.weight.detach(), x.detach()
        FPweight_1, FPweight_2, FPact_1, FPact_2 = self.select(FPweight, FPact)

        out = self.conv(FPweight_1, FPweight_2, FPact_1, FPact_2, quantized=False)
        index = self.act_bit - 2

        if self.quan_weight:

            if (self.weight_bit & 1) == 0:
                FPweight_1 = FPweight_1.chunk(self.weight_bit // 2, dim=0)
                for i in range(len(FPweight_1)):
                    self.uW[i].data.fill_(FPweight[i].std() * 3.0)
                    self.lW[i].data.fill_(-FPweight[i].std() * 3.0)

            else:
                FPweight_1 = FPweight_1.chunk(self.weight_bit // 2, dim=0)
                for i in range(len(FPweight_1)):
                    self.uW[i].data.fill_(FPweight[i].std() * 3.0)
                    self.lW[i].data.fill_(-FPweight[i].std() * 3.0)

                self.uW[self.weight_bit // 2].data.fill_(FPweight_2.std() * 3.0)
                self.lW[self.weight_bit // 2].data.fill_(-FPweight_2.std() * 3.0)


        if self.quan_act:

            if (self.weight_bit & 1) == 0:
                FPact_1 = FPact_1.chunk(self.weight_bit // 2, dim=1)
                for i in range(len(FPact_1)):
                    self.uA[i][index].data.fill_(FPact_1[i].std() / math.sqrt(1 - 2 / math.pi) * 3.0)
                    self.lA[i][index].data.fill_(FPact_1[i].min())

            else:
                FPact_1 = FPact_1.chunk(self.weight_bit // 2, dim=1)
                for i in range(len(FPact_1)):
                    self.uA[i][index].data.fill_(FPact_1[i].std() / math.sqrt(1 - 2 / math.pi) * 3.0)
                    self.lA[i][index].data.fill_(FPact_1[i].min())

                self.uA[self.weight_bit // 2][index].data.fill_(FPact_2.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
                self.lA[self.weight_bit // 2][index].data.fill_(FPact_2.min())

        FPweight, FPact = self.weight.detach(), x.detach()
        FPweight_1, FPweight_2, FPact_1, FPact_2 = self.select(FPweight, FPact)

        Qout = self.conv(FPweight_1, FPweight_2, FPact_1, FPact_2, quantized=True)

    def select(self, weight, act):

        if self.first_conv: # only select output channel
            index_o = self.out_channels // self.groups * (self.weight_bit // 2)
            index_i_fir = self.in_channels // self.groups * (self.weight_bit // 2)
            index_i_sec = self.in_channels // self.groups * ((self.weight_bit + 1) // 2)
            single_len = self.in_channels // self.groups

            # V2
            if self.weight_bit == 2 or self.weight_bit == 4:
                return weight[:index_o, :, :, :], None, act[:, :index_i_fir, :, :], None
            elif self.weight_bit == 3:
                return weight[:index_o, :, :, :], weight[index_o:index_o + self.oneBit_outchannel, :, :, :], \
                       act[:, :index_i_fir, :, :], act[:, index_i_fir:index_i_sec, :, :]

        elif self.last_conv: # only select input channel
            index_o = self.out_channels // self.groups * (self.weight_bit // 2)
            index_i_fir = self.in_channels // self.groups * (self.weight_bit // 2)
            index_i_sec = self.in_channels // self.groups * (self.weight_bit // 2) + self.oneBit_inchannel
            single_len = self.in_channels // self.groups
            # V2
            if self.weight_bit == 2 or self.weight_bit == 4:
                return weight[:index_o, :index_i_fir, :, :], None, act, None
            elif self.weight_bit == 3:
                return weight[:index_o, :, :, :], \
                       weight[index_o:index_o + self.out_channels // self.groups, :self.oneBit_inchannel, :, :], \
                       act[:, :index_i_fir, :, :], act[:, index_i_fir:index_i_sec, :, :]
        else:
            index_o = self.out_channels // self.groups * (self.weight_bit // 2)
            index_i_fir = self.in_channels // self.groups * (self.weight_bit // 2)
            index_i_sec = self.in_channels // self.groups * (self.weight_bit // 2) + self.oneBit_inchannel
            single_len = self.in_channels // self.groups
            # V2
            if self.weight_bit == 2 or self.weight_bit == 4:
                return weight[:index_o, :, :, :], None, act, None
            elif self.weight_bit == 3:
                return weight[:index_o, :, :, :], \
                       weight[index_o:index_o + self.oneBit_outchannel, :self.oneBit_inchannel, :, :], \
                       act[:, :index_i_fir, :, :], act[:, index_i_fir:index_i_sec, :, :]

    def conv(self, FPweight_1, FPweight_2, FPact_1, FPact_2, quantized=True):
        if self.first_conv:
            if FPweight_2 is None: # 2, 4, 6, 8
                if quantized:
                    Qweight_1 = self.group_weight_quantization(FPweight_1)
                    Qact_1 = self.group_activation_quantization(FPact_1)
                else:
                    Qweight_1, Qact_1 = FPweight_1, FPact_1

                output = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.weight_bit // 2) # self.weight_bit // 2 :-> groups
            else:
                if quantized:
                    Qweight_1, Qweight_2 = self.group_weight_quantization(FPweight_1, FPweight_2)
                    Qact_1, Qact_2 = self.group_activation_quantization(FPact_1, FPact_2)
                else:
                    Qweight_1, Qweight_2, Qact_1, Qact_2 = FPweight_1, FPweight_2, FPact_1, FPact_2

                output_1 = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.weight_bit // 2) # self.weight_bit // 2 :-> groups
                output_2 = F.conv2d(Qact_2, Qweight_2, self.bias, self.stride, self.padding, self.dilation,
                                    1) # 1 :-> groups
                output = torch.cat((output_1, output_2), dim=1)

        elif self.last_conv:
            if FPweight_2 is None:  # 2, 4, 6, 8
                if quantized:
                    Qweight_1 = self.group_weight_quantization(FPweight_1)
                    Qact_1 = self.group_activation_quantization(FPact_1)
                else:
                    Qweight_1, Qact_1 = FPweight_1, FPact_1

                output = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.weight_bit // 2) # self.weight_bit // 2 :-> groups
            else:
                if quantized:
                    Qweight_1, Qweight_2 = self.group_weight_quantization(FPweight_1, FPweight_2)
                    Qact_1, Qact_2 = self.group_activation_quantization(FPact_1, FPact_2)
                else:
                    Qweight_1, Qweight_2, Qact_1, Qact_2 = FPweight_1, FPweight_2, FPact_1, FPact_2

                output_1 = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.weight_bit // 2) # self.weight_bit // 2 :-> groups
                output_2 = F.conv2d(Qact_2, Qweight_2, self.bias, self.stride, self.padding, self.dilation,
                                    1) # 1 :-> groups
                output = torch.cat((output_1, output_2), dim=1)

        else:
            if FPweight_2 is None: # 2, 4, 6, 8
                if quantized:
                    Qweight_1 = self.group_weight_quantization(FPweight_1)
                    Qact_1 = self.group_activation_quantization(FPact_1)
                else:
                    Qweight_1, Qact_1 = FPweight_1, FPact_1
                output = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.weight_bit // 2) # self.weight_bit // 2 :-> groups
            else:
                if quantized:
                    Qweight_1, Qweight_2 = self.group_weight_quantization(FPweight_1, FPweight_2)
                    Qact_1, Qact_2 = self.group_activation_quantization(FPact_1, FPact_2)
                else:
                    Qweight_1, Qweight_2, Qact_1, Qact_2 = FPweight_1, FPweight_2, FPact_1, FPact_2

                output_1 = F.conv2d(Qact_1, Qweight_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.weight_bit // 2)# self.weight_bit // 2 :-> groups
                output_2 = F.conv2d(Qact_2, Qweight_2, self.bias, self.stride, self.padding, self.dilation,
                                    1) # 1 :-> groups
                output = torch.cat((output_1, output_2), dim=1)

        return output

    def forward(self, x):

        if self.init == 1:
            self.initialize(x)

        # if torch.any(torch.isnan(x)):
        #     print('2')
        #     import IPython
        #     IPython.embed()

        FPweight, FPact = self.weight, x
        FPweight_1, FPweight_2, FPact_1, FPact_2 = self.select(FPweight, FPact)

        # if FPweight_2 is not None:
        #     print(FPweight.shape, FPact.shape, FPweight_1.shape, FPweight_2.shape, FPact_1.shape, FPact_2.shape)
        # else:
        #     print(FPweight.shape, FPact.shape, FPweight_1.shape, FPact_1.shape)

        output = self.conv(FPweight_1, FPweight_2, FPact_1, FPact_2, quantized=True)



        return output



def QConv3x3(args, in_planes, out_planes, stride=1, groups=1, dilation=1, oneBit_outchannel=-1, oneBit_inchannel=-1,
             last_conv=False, first_conv=False):
    """3x3 convolution with padding"""
    return QConv(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False, dilation=dilation, args=args,
                 oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                 last_conv=last_conv, first_conv=first_conv)


def QConv1x1(args, in_planes, out_planes, stride=1, groups=1, oneBit_outchannel=-1, oneBit_inchannel=-1, last_conv=False):
    """1x1 convolution"""
    return QConv(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False, args=args,
                 oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel, last_conv=last_conv)


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, out_channels, oneBit_outchannel, groups=-1, last_conv=False):
        super(SwitchableBatchNorm2d, self).__init__()

        self.last_conv = last_conv
        self.groups = groups

        bns = []

        if self.last_conv:
            bns.append(nn.BatchNorm2d(out_channels // groups * 1))  # 2
            bns.append(nn.BatchNorm2d(out_channels // groups * 2))  # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 2))  # 4
            bns.append(nn.BatchNorm2d(out_channels // groups * 3))  # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 3))  # 6
            bns.append(nn.BatchNorm2d(out_channels // groups * 4))  # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 4))  # 8
            bns.append(nn.BatchNorm2d(out_channels))
        else:
            bns.append(nn.BatchNorm2d(out_channels // groups * 1)) # 2
            bns.append(nn.BatchNorm2d(out_channels // groups * 1 + oneBit_outchannel)) # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 2)) # 4
            bns.append(nn.BatchNorm2d(out_channels // groups * 2 + oneBit_outchannel)) # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 3)) # 6
            bns.append(nn.BatchNorm2d(out_channels // groups * 3 + oneBit_outchannel)) # 3
            bns.append(nn.BatchNorm2d(out_channels // groups * 4)) # 8
        self.bn = nn.ModuleList(bns)
        self.weight_bit = 2
        self.act_bit = 2


    def forward(self, x):
        index = self.weight_bit-2
        output = self.bn[index](x)

        return output

# class SwitchableBatchNorm2d(nn.Module):
#     def __init__(self, out_channels, oneBit_outchannel, groups=-1, last_conv=False):
#         super(SwitchableBatchNorm2d, self).__init__()
#
#         self.last_conv = last_conv
#         self.groups = groups
#         self.weight_bit = 2
#         self.act_bit = 2
#
#
#         bns_2bit = []
#         for _ in range(7):
#             if self.last_conv:
#                 bns_2bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_2bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#         self.bns_2bit = nn.ModuleList(bns_2bit)
#
#         bns_3bit = []
#         for _ in range(6):
#             if self.last_conv:
#                 bns_3bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_3bit.append(nn.BatchNorm2d(oneBit_outchannel))
#         self.bns_3bit = nn.ModuleList(bns_3bit)
#
#         bns_4bit = []
#         for _ in range(5):
#             if self.last_conv:
#                 bns_4bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_4bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#         self.bns_4bit = nn.ModuleList(bns_4bit)
#
#         bns_5bit = []
#         for _ in range(4):
#             if self.last_conv:
#                 bns_5bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_5bit.append(nn.BatchNorm2d(oneBit_outchannel))
#         self.bns_5bit = nn.ModuleList(bns_5bit)
#
#
#         bns_6bit = []
#         for _ in range(3):
#             if self.last_conv:
#                 bns_6bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_6bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#         self.bns_6bit = nn.ModuleList(bns_6bit)
#
#         bns_7bit = []
#         for _ in range(2):
#             if self.last_conv:
#                 bns_7bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_7bit.append(nn.BatchNorm2d(oneBit_outchannel))
#         self.bns_7bit = nn.ModuleList(bns_7bit)
#
#         bns_8bit = []
#         for _ in range(1):
#             if self.last_conv:
#                 bns_8bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#             else:
#                 bns_8bit.append(nn.BatchNorm2d(out_channels // groups * 1))
#         self.bns_8bit = nn.ModuleList(bns_8bit)
#         self.weight_bit = 8
#
#     def branch_bn(self, inp, group_index, double=True):
#
#         if double:
#             if group_index == 0: # 1st branch
#                 group_index = 0
#                 index = self.act_bit - 2
#             elif group_index == 1: # 2st branch
#                 group_index = 2
#                 index = self.act_bit - 4
#             elif group_index == 2: # 3st branch
#                 group_index = 4
#                 index = self.act_bit - 6
#             elif group_index == 3: # 4st branch
#                 group_index = 6
#                 index = self.act_bit - 8
#         else:
#             if group_index == 1: # 2st branch
#                 group_index = 1
#                 index = self.act_bit - 3
#             elif group_index == 2: # 3st branch
#                 group_index = 3
#                 index = self.act_bit - 5
#             elif group_index == 3: # 4st branch
#                 group_index = 5
#                 index = self.act_bit - 7
#
#         if group_index == 0:
#             inp = self.bns_2bit[index](inp)
#         elif group_index == 1:
#             inp = self.bns_3bit[index](inp)
#         elif group_index == 2:
#             inp = self.bns_4bit[index](inp)
#         elif group_index == 3:
#             inp = self.bns_5bit[index](inp)
#         elif group_index == 4:
#             inp = self.bns_6bit[index](inp)
#         elif group_index == 5:
#             inp = self.bns_7bit[index](inp)
#         elif group_index == 6:
#             inp = self.bns_8bit[index](inp)
#         return inp
#
#     def forward(self, x):
#
#         if self.weight_bit == 2 or self.weight_bit == 4 or self.weight_bit == 6 or self.weight_bit == 8:
#             tmp_list = []
#             x = x.chunk(self.weight_bit // 2, dim=1)
#             for i in range(len(x)):
#                 tmp = self.branch_bn(x[i], i)
#                 tmp_list.append(tmp)
#             output = torch.cat(tmp_list, dim=1)
#
#         if self.weight_bit == 3 or self.weight_bit == 5 or self.weight_bit == 7:
#             tmp_list = []
#             x = x.chunk((self.weight_bit+1) // 2, dim=1)
#             for i in range(len(x)):
#                 tmp = self.branch_bn(x[i], i, double=False)
#                 tmp_list.append(tmp)
#             output = torch.cat(tmp_list, dim=1)
#
#         return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, quan_weight, quan_act, weight_levels, act_levels, bias=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quan_weight = quan_weight
        self.quan_act = quan_act
        self.STE_discretizer = STE_discretizer.apply

        if self.quan_weight:
            self.weight_levels = 2 ** 8
            self.uW = nn.Parameter(data=torch.tensor(0).float())
            self.lW = nn.Parameter(data=torch.tensor(0).float())

        if self.quan_act:
            self.act_levels = 2 ** 8
            self.uA = nn.Parameter(data=torch.tensor(0).float())
            self.lA = nn.Parameter(data=torch.tensor(0).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(data=torch.tensor(1).float())

    def weight_quantization(self, weight):
        if not self.quan_weight:
            return weight
        weight = (weight - self.lW) / (self.uW - self.lW)
        weight = weight.clamp(min=0, max=1)  # [0, 1]
        weight = self.STE_discretizer(weight, self.weight_levels)
        weight = (weight - 0.5) * 2  # [-1, 1]
        return weight

    def act_quantization(self, x):
        if not self.quan_act:
            return x
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1)  # [0, 1]
        x = self.STE_discretizer(x, self.act_levels)
        return x

    def initialize(self, x):
        # self.init.data.fill_(0)
        Qweight = self.weight
        Qact = x

        if self.quan_weight:
            self.uW.data.fill_(self.weight.std() * 3.0)
            self.lW.data.fill_(-self.weight.std() * 3.0)
            Qweight = self.weight_quantization(self.weight)

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self.act_quantization(x)

        Qout = F.linear(Qact, Qweight, self.bias)
        out = F.linear(x, self.weight, self.bias)
        self.output_scale.data.fill_(out.abs().mean() / Qout.abs().mean())

    def forward(self, x):
        if self.init == 1:
            self.initialize(x)

        Qweight = self.weight
        if self.quan_weight:
            Qweight = self.weight_quantization(Qweight)

        Qact = x
        if self.quan_act:
            Qact = self.act_quantization(Qact)

        output = F.linear(Qact, Qweight, self.bias) * torch.abs(self.output_scale)

        return output