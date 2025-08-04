import random
import os
import sys
import pickle

import timm

import torch
import torchvision
from torchvision.transforms import v2

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tqdm

import utils.self_utils as self_utils
import utils.cubDataset as CUB

import MSA_CC


dataset_path = './CUB-200-2011/CUB_200_2011/'
image_path = os.path.join(dataset_path + '/images/')


device = None
seed = 0


# Set all random seeds to desired seed
def set_seeds(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    _ = torch.manual_seed(SEED)
    seed = SEED


# Get CUDA devices
def setup_cuda():
    if torch.cuda.is_available:
        device = 'cuda:0'
        print('CUDA enabled!')
    else:
        device = 'cpu'
        print('CPU enabled.')

    return device


# Defining some functional transformations as "compiled" transformations 
class CenterCrop(torch.nn.Module):
    def __init__(self, percent=0.8):
        super().__init__()
        self.frac = percent

    def forward(self, img):
        return v2.functional.center_crop(img, (int(self.frac*img.size[1]), int(self.frac*img.size[0])))


class rot90(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return v2.functional.rotate(img, 90)


def load_original_data(batch_size, ds=False):
    train_data = CUB.cubDataset(dataset_path, image_path, train=True)
    test_data = CUB.cubDataset(dataset_path, image_path, train=False)

    train_size = train_data.length
    test_size = test_data.length

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True) 
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True) 
    
    if ds:
        return train_dataloader, test_dataloader, train_size, test_size, train_data, test_data
    else:
        return train_dataloader, test_dataloader, train_size, test_size


def load_model(pretrain=True, mtype='21k_b_big', cumulative=True):
    if pretrain and mtype == '21k_b_big':
        backbone = torch.load('./Pretrained/pretrained_swinv2_b_384.pt', weights_only=False)
        bbo = 1920
    elif mtype == 'timm':
        backbone = timm.create_model('swinv2_large_window12to24_192to384.ms_in22k_ft_in1k', pretrained=True)
        bbo = 2880

    for param in backbone.parameters():
        param.requires_grad = False

    for name, param in backbone.named_parameters():
        if 'head' in name: param.requires_grad = True
        if 'layers.3' in name: param.requires_grad = True
        if 'layers.2' in name: param.requires_grad = True
        if 'layers.1' in name: param.requires_grad = True
        if 'layers.0' in name: param.requires_grad = True
        if name == 'norm.bias': param.requires_grad = True
        if name == 'norm.weight': param.requires_grad = True

    model = MSA_CC.MSA(backbone=backbone, backbone_out=bbo, single=True)

    return model


if __name__ == '__main__':
    _, _, _, _, train, _ = load_original_data(batch_size=1, ds=True)
    print(train.__getitem__(0))
    get_class_freqs(train)

    #model = load_model()
