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
import utils.nabirds as nabirds
import utils.nabirdsDataset as NBD

import MSA_CC


dataset_path = './nabirds-data/nabirds/'
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


# Loading mappings from numerical labels to human-interpretable labels
def load_label_helpers():
    hierarchical_labels = {}
    with open('./labels/hierarchical_class_labels.pkl', 'rb') as f:
        hierarchical_labels = pickle.load(f)

    normalized_labels = []
    with open('./labels/normalized_class_labels_fixed.pkl', 'rb') as f:
        normalized_labels = pickle.load(f)

    normalized_class_names = []
    with open('./labels/normalized_class_mappings.pkl', 'rb') as f:
        normalized_class_names = pickle.load(f)

    """
    Given the following:
        label = normalized_labels[420]
        
    > print(label)
[   1  44 323 428]

    > print(normalized_class_names[0][label[0]])
    Perching Birds

    > print(normalized_class_names[1][label[1]])
    Wood-Warblers

    > print(normalized_class_names[2][label[2]])
    Palm Warbler

    > print(normalized_class_names[3][label[3]])
    Palm Warbler
    """

    return hierarchical_labels, normalized_labels, normalized_class_names


def get_class_freqs(ds):
    c1 = torch.zeros(2)
    c2 = torch.zeros(50)
    c3 = torch.zeros(404)
    c4 = torch.zeros(555)

    _, labels, _ = load_label_helpers()

    for i in range(ds.__len__()):
        _, label = ds.__getitem__(i)
        real_label = labels[label]

        c1[real_label[0]] += 1
        c2[real_label[1]] += 1
        c3[real_label[2]] += 1
        c4[real_label[3]] += 1

    
    c1 = ds.length / (c1 * 2)
    c2 = ds.length / (c2 * 50)
    c3 = ds.length / (c3 * 404)
    c4 = ds.length / (c4 * 555)

    
    with open('./labels/c1_freq_orig.pkl', 'wb') as f:
        pickle.dump(c1,f)

    with open('./labels/c2_freq_orig.pkl', 'wb') as f:
        pickle.dump(c2,f)

    with open('./labels/c3_freq_orig.pkl', 'wb') as f:
        pickle.dump(c3,f)

    with open('./labels/c4_freq_orig.pkl', 'wb') as f:
        pickle.dump(c4,f)


# Load class frequencies
def load_orig_class_freqs():
    c1_freq = []
    with open('./labels/c1_freq_orig.pkl', 'rb') as f:
        c1_freq = torch.tensor(pickle.load(f))

    c2_freq = []
    with open('./labels/c2_freq_orig.pkl', 'rb') as f:
        c2_freq = torch.tensor(pickle.load(f))

    c3_freq = []
    with open('./labels/c3_freq_orig.pkl', 'rb') as f:
        c3_freq = torch.tensor(pickle.load(f))

    c4_freq = []
    with open('./labels/c4_freq_orig.pkl', 'rb') as f:
        c4_freq = torch.tensor(pickle.load(f))

    return c1_freq, c2_freq, c3_freq, c4_freq


# Load class frequencies
def load_class_freqs():
    c1_freq = []
    with open('./labels/c1_freq.pkl', 'rb') as f:
        c1_freq = torch.tensor(pickle.load(f))

    c2_freq = []
    with open('./labels/c2_freq.pkl', 'rb') as f:
        c2_freq = torch.tensor(pickle.load(f))

    c3_freq = []
    with open('./labels/c3_freq.pkl', 'rb') as f:
        c3_freq = torch.tensor(pickle.load(f))

    c4_freq = []
    with open('./labels/c4_freq.pkl', 'rb') as f:
        c4_freq = torch.tensor(pickle.load(f))

    return c1_freq, c2_freq, c3_freq, c4_freq


# Load dataset, returning as dataloaders
def load_data(split=0.5, length=48562, batch_size=2):
    trans = v2.Compose([      #v2.RandomRotation(180),
                              #CenterCrop(),
                              #v2.RandomResizedCrop(224),
                              v2.Resize((384,384)),
                              rot90(),
                              #v2.RandomHorizontalFlip(),
                              v2.PILToTensor(),
                              v2.ConvertImageDtype(torch.float),
                              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_size = int((1-split)*length)
    test_size = length - train_size

    full_dataset = torchvision.datasets.ImageFolder(image_path, transform=trans)
    ds_train_idx, ds_test_idx = train_test_split(list(range(len(full_dataset))), test_size=split, stratify=full_dataset.targets, random_state=seed)
    ds_train = torch.utils.data.Subset(full_dataset, ds_train_idx)
    ds_test = torch.utils.data.Subset(full_dataset, ds_test_idx)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True)

    return train_dataloader, test_dataloader, train_size, test_size


def load_original_data(batch_size, ds=False):
    #h = nabirds.load_hierarchy(dataset_path)
    #print(h)
    _, normalized_labels, names = load_label_helpers()

    train, test = nabirds.load_train_test_split(dataset_path)
    
    train_data = NBD.nabirdsDataset(dataset_path, image_path, general=False, ignore=train, normalized_names=names)
    test_data = NBD.nabirdsDataset(dataset_path, image_path, general=False, ignore=test, normalized_names=names)

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

    model = MSA_CC.MSA(backbone=backbone, backbone_out=bbo)

    return model


if __name__ == '__main__':
    _, _, _, _, train, _ = load_original_data(batch_size=1, ds=True)
    print(train.__getitem__(0))
    get_class_freqs(train)

    #model = load_model()
