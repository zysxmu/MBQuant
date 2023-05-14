# ref. https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
# from ast import arg
import os
import random
import shutil
import time
import warnings
import pdb
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from model_cifar.quant_resnet4cifar_my import *
from model_cifar.quant_conv import QConv
# from model_cifar.quant_resnet4cifar_my_234 import *
# from model_cifar.quant_conv_234 import QConv
from utils.utils import save_checkpoint, accuracy, AverageMeter, ProgressMeter, Time2Str, setup_logging, CrossEntropyLossSoft
from option import args

best_acc1 = 0
np.random.seed(0)


def main():
    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    args.log_dir += '/' + args.datasetsname  + '_' + args.arch  + '_' + ''.join(args.bit_list) + '/' + Time2Str()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def init_quant_model(model, args, ngpus_per_node):
    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(1)

    data_means = {
        'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
        'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]],
    }
    data_stds = {
        'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
        'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]],
    }
    if args.datasetsname == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar10'], data_stds['cifar10'])
        ])
        train_data = datasets.CIFAR10(root=args.data,
                                      train=True,
                                      transform=train_transform,
                                      download=True)
        train_data.num_classes = 10

    elif args.datasetsname == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar100'], data_stds['cifar100'])
        ])
        train_data = datasets.CIFAR100(root=args.data,
                                       train=True,
                                       transform=train_transform,
                                       download=True)
        train_data.num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=int(args.batch_size / ngpus_per_node), shuffle=True,
        num_workers=int((args.workers + ngpus_per_node - 1) / ngpus_per_node), pin_memory=True)
    iterloader = iter(train_loader)

    model.to(args.gpu)
    model.train()
    for i in range(len(args.bit_list)):
        print('init for bit-width: ', args.bit_list[i])
        images, labels = next(iterloader)
        images = images.to(args.gpu)
        labels = labels.to(args.gpu)
        for name, layers in model.named_modules():
            if hasattr(layers, 'act_bit'):
                setattr(layers, "act_bit", int(args.bit_list[i]))
            if hasattr(layers, 'weight_bit'):
                setattr(layers, "weight_bit", int(args.bit_list[i]))
        model.forward(images)

    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(0)


def cal_params(model, args):
    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []

    for name, layers in model.named_modules():
        if isinstance(layers, QConv):
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)
            if layers.quan_weight:
                if isinstance(layers.lW, nn.ParameterList):
                    for x in layers.lW:
                        quant_params.append(x)
                    for x in layers.uW:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lW)
                    quant_params.append(layers.uW)
            if layers.quan_act:
                if isinstance(layers.lA, nn.ParameterList):
                    for x in layers.lA:
                        quant_params.append(x)
                    for x in layers.uA:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lA)
                    quant_params.append(layers.uA)
            if layers.quan_act or layers.quan_weight:
                if hasattr(layers, 'output_scale'):
                    if isinstance(layers.output_scale, nn.ParameterList):
                        for x in layers.output_scale:
                            quant_params.append(x)
                    else:
                        quant_params.append(layers.output_scale)

        elif isinstance(layers, nn.Conv2d): # first conv xxxx
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)

        elif isinstance(layers, nn.Linear): # last FC xxxx
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)

        elif isinstance(layers, nn.SyncBatchNorm) or isinstance(layers, nn.BatchNorm2d):
            if layers.bias is not None:
                model_params.append(layers.weight)
                model_params.append(layers.bias)

    log_string = "total params: {}, trainable model params: {}, trainable quantizer params: {}".format(
        sum(p.numel() for p in trainable_params), sum(p.numel() for p in model_params),
        sum(p.numel() for p in quant_params)
    )
    assert sum(p.numel() for p in trainable_params) == (sum(p.numel() for p in model_params)
                                                        + sum(p.numel() for p in quant_params))
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
        logging.info(log_string)

    return model_params, quant_params


def loading_data(args):
    data_means = {
        'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
        'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]],
    }
    data_stds = {
        'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
        'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]],
    }

    if args.datasetsname == 'cifar10':
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(data_means['cifar10'], data_stds['cifar10'])
            ])
        train_dataset = datasets.CIFAR10(root=args.data,
                                    train=True,
                                    transform=train_transform,
                                    download=True)
        train_dataset.num_classes = 10

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar10'], data_stds['cifar10'])
        ])
        val_dataset = datasets.CIFAR10(root=args.data,
                                      train=False,
                                      transform=val_transform)

    elif args.datasetsname == 'cifar100':
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(data_means['cifar100'], data_stds['cifar100'])
            ])
        train_dataset = datasets.CIFAR100(root=args.data,
                                    train=True,
                                    transform=train_transform,
                                    download=True)
        train_dataset.num_classes = 100

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_means['cifar100'], data_stds['cifar100'])
        ])
        val_dataset = datasets.CIFAR100(root=args.data,
                                      train=False,
                                      transform=val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu == 0:
        setup_logging(os.path.join(args.log_dir, "log.txt"))
        arg_dict = vars(args)
        log_string = 'configs\n'
        for k, v in arg_dict.items():
            log_string += "{}: {}\n".format(k, v)
        logging.info(log_string)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    if args.datasetsname == 'cifar10':
        args.num_classes = 10
    if args.datasetsname == 'cifar100':
        args.num_classes = 100
    # create model
    model_class = globals().get(args.arch)
    model = model_class(args, groups=args.groups, num_classes=args.num_classes)
    # model = model_class(args, num_classes=args.num_classes)


    args.bit_list = [eval(x) for x in args.bit_list]
    if args.rank % ngpus_per_node == 0:
        # print(model)
        logging.info('args.bit_list: {}'.format(args.bit_list))

    ### initialze quantizer parameters
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):  # do only at rank=0 process
        if not args.evaluate:
            init_quant_model(model, args, ngpus_per_node)
    ###
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    train_loader, val_loader, train_sampler = loading_data(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.evaluate:
        if os.path.exists(args.model):
            model_dict = torch.load(args.model)
            if 'state_dict' in model_dict:
                model_dict = model_dict['state_dict']
            model.load_state_dict(model_dict)
            validate(val_loader, model, criterion, args, None, args.start_epoch)
        else:
            raise ValueError("model path {} not exists".format(args.model))
        return

    model_params, quant_params = cal_params(model, args)
    print('len(model_params)', len(model_params))

    if args.optimizer_m == 'SGD':
        optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum,
                                      weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer_m == 'Adam':
        optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    if args.lr_scheduler == 'cosine':
        scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)
    elif args.lr_scheduler == 'step':
        if args.decay_schedule is not None:
            milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
        else:
            milestones = [(args.epochs + 1)]
        scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones, gamma=args.gamma)
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones, gamma=args.gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            optimizer_q.load_state_dict(checkpoint['optimizer_q'])
            scheduler_m.load_state_dict(checkpoint['scheduler_m'])
            scheduler_q.load_state_dict(checkpoint['scheduler_q'])
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    ### tensorboard
    # if args.rank % ngpus_per_node == 0: # do only at rank=0 process
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
    ###

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m, scheduler_q, epoch, args, writer)

        # evaluate all bits on validation set
        acc1 = np.zeros(len(args.bit_list))
        for i in range(0, len(args.bit_list)):
            for name, layers in model.named_modules():
                # if hasattr(layers, 'cur_bit'):
                #     setattr(layers, "cur_bit", int(args.bit_list[i]))
                if hasattr(layers, 'act_bit'):
                    setattr(layers, "act_bit", int(args.bit_list[i]))
                if hasattr(layers, 'weight_bit'):
                    setattr(layers, "weight_bit", int(args.bit_list[i]))
            acc1[i] = validate(val_loader, model, criterion, args, writer, epoch, weight_bit=int(args.bit_list[i]),
                               act_bit=int(args.bit_list[i]))
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("act_bit: {}, weight_bit: {}, acc1: {}".format(int(args.bit_list[i]), int(args.bit_list[i]), acc1[i]))

        acc1_avg = np.mean(acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
            logging.info("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "current avg acc@1: {}".format(acc1_avg))
        is_best = acc1_avg > best_acc1
        best_acc1 = max(acc1_avg, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'best_acc1': best_acc1,
                'optimizer_m': optimizer_m.state_dict(),
                'optimizer_q': optimizer_q.state_dict(),
                'scheduler_m': scheduler_m.state_dict(),
                'scheduler_q': scheduler_q.state_dict(),
            }, is_best, path=args.log_dir)

        log_string = "Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + \
                     ' 1 epoch spends {:2d}:{:.2f} mins\t remain {:2d}:{:2d} hours\n'. \
                         format(int((time.time() - start) // 60), (time.time() - start) % 60,
                                int((args.epochs - epoch - 1) * (time.time() - start) // 3600),
                                int((args.epochs - epoch - 1) * (time.time() - start) % 3600 / 60))
        for i in range(0, len(args.bit_list)):
            log_string += '[{}bit] {:.3f}\t'.format(args.bit_list[i], acc1[i])
        log_string += "current best acc@1: {}\n".format(best_acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
            logging.info(log_string)

    if writer is not None:
        writer.close()


def train(train_loader, model, criterion, optimizer_m, optimizer_q, scheduler_m, scheduler_q, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = {}
    top1 = {}
    top5 = {}
    for bit in args.bit_list:
        losses[int(bit)] = AverageMeter('Loss', ':.4e')
        top1[int(bit)] = AverageMeter('Acc@1', ':6.2f')
        top5[int(bit)] = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        args=args,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # inplace distillation
    soft_criterion = CrossEntropyLossSoft(reduction='none').cuda()
    if len(args.bit_list) == 1:
        bits_train = [args.bit_list[0]]
    elif len(args.bit_list) == 2:
        bits_train = [args.bit_list[1], args.bit_list[0]]
    else:
        bits_train = args.bit_list[::-1]

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        optimizer_m.zero_grad()
        optimizer_q.zero_grad()
        for bit in bits_train:
            for name, layers in model.named_modules():
                if hasattr(layers, 'act_bit'):
                    setattr(layers, "act_bit", int(bit))
                if hasattr(layers, 'weight_bit'):
                    setattr(layers, "weight_bit", int(bit))
            # compute output
            output = model(images)

            # inplace distillation
            if bit == int(args.bit_list[-1]):
                loss = criterion(output, target)
                output_teacher_biggest_bit = output.detach()
            else:
                loss = (torch.mean(
                    soft_criterion(output, torch.nn.functional.softmax(
                        output_teacher_biggest_bit, dim=1))) + criterion(output, target))/2
            # loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses[bit].update(loss.item(), images.size(0))
            top1[bit].update(acc1[0], images.size(0))
            top5[bit].update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            loss.backward()

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                    logging.info("iter: {}, loss: {}, bit: {}".format(i, loss.item(), bit))

        optimizer_m.step()
        optimizer_q.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if writer is not None:  # this only works at rank=0 process
                writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], len(train_loader) * epoch + i)
                writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], len(train_loader) * epoch + i)
                writer.add_scalar('train/loss(current)', loss.cpu().item(), len(train_loader) * epoch + i)
                # writer.add_scalar('train/loss(average)', losses.avg, len(train_loader)*epoch + i)
                # writer.add_scalar('train/top1(average)', top1.avg, len(train_loader)*epoch + i)
                # writer.add_scalar('train/top5(average)', top5.avg, len(train_loader)*epoch + i)
            # progress.display(i)
    scheduler_m.step()
    scheduler_q.step()


def validate(val_loader, model, criterion, args, writer, epoch, weight_bit, act_bit):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        args=args,
        prefix="Epoch: [{}]".format(epoch) + '[Weight {}bit] Test: '.format(weight_bit)
               + '[Act {}bit] Test: '.format(act_bit))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if writer is not None:
            writer.add_scalar('val/top1_W{}bitA{}bit'.format(weight_bit, act_bit), top1.avg, epoch)
            writer.add_scalar('val/top1_W{}bitA{}bit'.format(weight_bit, act_bit), top5.avg, epoch)

        log_string = "Epoch: [{}]".format(epoch) + '[W{}bitA{}bit]'.format(weight_bit, act_bit) + "[GPU{}]".format(
            args.gpu) + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
            logging.info(log_string)

    return top1.avg


if __name__ == '__main__':
    main()
