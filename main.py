import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse
#Downloading the data
from torchvision.datasets import CIFAR100

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--epoch', default = 2, type = int) #default = 250
parser.add_argument('--lr', default = 0.1, type = int)
parser.add_argument('--max_size', default = 2000, type = int)
parser.add_argument('--total_cls', default = 100, type = int)
parser.add_argument('--path_to_train_dir', default = './data/cifar-100-python/train', type = str)
parser.add_argument('--path_to_test_dir', default = './data/cifar-100-python/test', type = str)

args = parser.parse_args()


#To download the CIFAR100 data from PyTorch 
if args.path_to_train_dir is './data/cifar-100-python/train':
    train_data = CIFAR100(download=True,root="./data")
if args.path_to_test_dir is './data/cifar-100-python/test':
    test_data = CIFAR100(root="./data",train=False)
print(f'train_data: \n{train_data} \n\n test_data: \n{test_data}')


if __name__ == "__main__":
    # showGod()
    trainer = Trainer(args.total_cls, args.path_to_train_dir, args.path_to_test_dir)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
