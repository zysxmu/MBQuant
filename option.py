import argparse
from utils.utils import str2bool

parser = argparse.ArgumentParser(description='PyTorch Implementation of MultiQuant')

# data and model
parser.add_argument('--name', type=str, default='This is a experiments', help='experiments name')
parser.add_argument('--data', metavar='DIR', default='/path/to/ILSVRC2012', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_quant', choices=('resnet18_quant',
                                                                                       'resnet34_quant',
                                                                                       'resnet50_quant',
                                                                                       'mv1_quant',
                                                                                       'mv2_quant',
                                                                                       'resnet8_quant',
                                                                                       'resnet20_quant',
                                                                                       'vgg11_bn_quant'), help='model architecture')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers')


# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer_m', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--lr_m', type=float, default=2.56e-2, help='learning rate for model parameters')
parser.add_argument('--lr_q', type=float, default=1e-4, help='learning rate for quantizer parameters')
parser.add_argument('--lr_m_end', type=float, default=0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--lr_q_end', type=float, default=0, help='final learning rate for quantizer parameters (for cosine)')
parser.add_argument('--decay_schedule', type=str, default='40-80', help='learning rate decaying schedule (for step)')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=str2bool)
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--pretrained', dest='pretrained', type=str2bool, default=True, help='use pre-trained model')

parser.add_argument('--model', dest='model', type=str,default=None, help=' model to eval')
parser.add_argument('--groups', default=4, type=int, help='groups to use.')


# misc & distributed data parallel
parser.add_argument('-p', '--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', type=str2bool, default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


# arguments for quantization
parser.add_argument('--bit_list', type=list, default="2345678", help='weight and activation quantization bit list of model. Expected input like: 2345 or 3456')
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
parser.add_argument('--weight_bit', type=int, default=2, help='weight quantization bit')
parser.add_argument('--act_bit', type=int, default=8, help='activation quantization bit')


parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')

parser.add_argument('--visible_gpus', default=None, type=str, help='total GPUs to use')


# logging
parser.add_argument('--log_dir', type=str, default='./results/TEST_ResNet18/')

parser.add_argument('--datasetsname', type=str, default='ImageNet')


args = parser.parse_args()
