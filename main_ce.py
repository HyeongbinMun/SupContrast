from __future__ import print_function

import os
import sys
import argparse
import time
import math
import torch
import wandb
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_path', type=str, default='',
                        help='path to custom dataset when dataset is "path"')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size for custom dataset or cifar upscaling')
    parser.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406],
                        help='mean for normalization when dataset is "path"')
    parser.add_argument('--std',  nargs=3, type=float, default=[0.229, 0.224, 0.225],
                        help='std for normalization when dataset is "path"')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '/ssd/hbmun/supcon/'
    opt.model_path = '/ssd/hbmun/supcon/models/path_models/{}_models'.format(opt.dataset)
    opt.tb_path = '/ssd/hbmun/supcon/models/path_models/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        if not opt.data_path:
            raise ValueError('When dataset is "path", you must set --data_path to your dataset root.')
        if not os.path.isdir(opt.data_path):
            raise ValueError(f'Invalid --data_path: {opt.data_path}')
        tr_dir = os.path.join(opt.data_path, 'train')
        va_dir = os.path.join(opt.data_path, 'val')
        if not (os.path.isdir(tr_dir) and os.path.isdir(va_dir)):
            raise ValueError(f'Expected subfolders "train" and "val" inside {opt.data_path}')
        opt.data_folder = opt.data_path
        opt.n_cls = None
    else:
        raise ValueError(f'dataset not supported: {opt.dataset}')

    return opt


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch
from collections import Counter

def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        if len(opt.mean) != 3 or len(opt.std) != 3:
            raise ValueError("For custom dataset, provide --mean and --std with 3 values each.")
        mean = tuple(opt.mean)
        std  = tuple(opt.std)
    else:
        raise ValueError(f'dataset not supported: {opt.dataset}')

    normalize = transforms.Normalize(mean=mean, std=std)
    img_size = opt.img_size

    # resize_blocks = [transforms.Resize((img_size, img_size))]
    resize_blocks = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]

    train_transform = transforms.Compose(
        resize_blocks + [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        resize_blocks + [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 3) dataset
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset   = datasets.CIFAR10(root=opt.data_folder,
                                         train=False,
                                         transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset   = datasets.CIFAR100(root=opt.data_folder,
                                          train=False,
                                          transform=val_transform)
    elif opt.dataset == 'path':
        train_dir = os.path.join(opt.data_folder, 'train')
        val_dir   = os.path.join(opt.data_folder, 'val')
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_transform)
        opt.n_cls = len(train_dataset.classes)

        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "class_to_idx mismatch between train and val.\n"
                f"train classes: {sorted(train_dataset.class_to_idx.keys())}\n"
                f"val   classes: {sorted(val_dataset.class_to_idx.keys())}\n"
                "-> Class Error"
            )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=pin,
        persistent_workers=opt.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=pin,
        persistent_workers=opt.num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader



def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        num_classes = output.size(1)
        k_list = (1, 5) if num_classes >= 5 else (1,)
        accs = accuracy(output, labels, topk=k_list)
        acc1 = accs[0]
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            num_classes = output.size(1)
            k_list = (1, 5) if num_classes >= 5 else (1,)
            accs = accuracy(output, labels, topk=k_list)
            acc1 = accs[0]
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # wandb init
    wandb.init(
        project="SupCon",
        name=f"{opt.model_name}",
        notes="CrossEntropy Training with ResNet and Adult",
        tags=["SupCon", opt.dataset, opt.model],
        config=vars(opt)
    )

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluation
        val_loss, val_acc = validate(val_loader, model, criterion, opt)

        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
