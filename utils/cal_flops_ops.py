import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    # print("count convNd.")
    # print(m)
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    if hasattr(m, 'act_levels'):
        # print("m True.")
        m.total_ops += calculate_conv2d_flops(
            input_size = list(x.shape),
            output_size = list(y.shape),
            kernel_size = list(m.weight.shape),
            groups = m.groups,
            bias = m.bias
        )
    # else:
    #     print("m False.")
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )

def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

# nn.Linear
def count_linear(m, x, y):
    # print("count linear.")
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)
