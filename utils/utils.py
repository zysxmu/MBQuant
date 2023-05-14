import os
import time
import argparse
import logging
import numpy as np

from thop import profile
from thop import clever_format
from torchinfo import summary
import shutil
from utils.cal_flops_ops import *
import torch


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Time2Str():
    sec = time.time()
    tm = time.localtime(sec)
    time_str = '{}{:02d}{:02d}_{:02d}_{:02d}'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min)
    return time_str


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def cal_flops(model, logging):
    # thop
    myinput = torch.randn(1, 3, 224, 224)
    macs, params, ret_dict = profile(model, inputs=(myinput, ), ret_layer_info=True, report_missing=True, custom_ops={QConv: count_convNd, QLinear: count_linear})
    logging.info("Model FLOPs and params: " + str(macs) + "\t " + str(params))
    macs, params = clever_format([macs, params], "%.3f")
    logging.info("Model FLOPs and params: " + str(macs) + "\t " + str(params))


def save_checkpoint(state, is_best, path='./', filename=''):
    ckptname = 'checkpoint' + filename + '.pth.tar'
    torch.save(state, os.path.join(path, ckptname))
    if is_best:
        bestname = 'model_best' + filename + '.pth.tar'
        shutil.copyfile(os.path.join(path, ckptname), os.path.join(path, bestname))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, args, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.args = args

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_string = '\t'.join(entries)
        if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed and self.args.rank  == 0):
            logging.info(log_string)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

