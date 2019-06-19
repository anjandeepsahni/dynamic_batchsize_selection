import os
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 2
DEFAULT_MODEL_PATH = './../Models'
DEFAULT_DATA_PATH = './../Data'
DEFAULT_CKPT = 'model_20190618-180049_e100_val_82.262.pt'
DEFAULT_RANDOM_SEED = 2222
DEFAULT_TRANSFORMS = transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])

def build_cifar10(root='./data', transform=DEFAULT_TRANSFORMS, batch_size=4, num_workers=2):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers,
                                              drop_last=True)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            drop_last=True)
    return trainset, testset, trainloader, testloader

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Size Variation Experiments')
    parser.add_argument('--lr', default=DEFAULT_LEARNING_RATE, type=float, help='learning rate')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, type=str, help='checkpoint file name')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str, help='model/checkpoint dir path')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH, type=str, help='model/checkpoint dir path')
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--num_workers', default=DEFAULT_NUM_WORKERS, type=int, help='number of worker threads')
    parser.add_argument('--random_seed', default=DEFAULT_RANDOM_SEED, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=DEFAULT_WEIGHT_DECAY, type=int, help='weight decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# as per alexnet
def init_weights(model):
    conv_count = 0
    linear_count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if conv_count in [1,3,4]:
                m.bias.data.fill_(1)
            else:
                m.bias.data.zero_()
            conv_count += 1
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if linear_count < 2:
                m.bias.data.fill_(1)
            else:
                m.bias.data.zero_()
            linear_count += 1
    return model
