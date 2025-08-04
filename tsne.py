import random
import os
import sys

# Issues with timm, ttach, and pytorch grad cam importing on one of our machines, so we used manual installations, shouldn't be necessary for most
#sys.path.append('./timm_utils/pytorch-image-models')
import timm
#sys.path.append('./ttach')
import ttach
#sys.path.append('./pytorch-grad-cam')
import pytorch_grad_cam

import torch
import torchvision
from torchvision.transforms import v2

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tqdm

import utils.self_utils as self_utils


TRITON_PTXAS_PATH="/usr/local/cuda-11.8/bin/ptxas"

dataset_path = './nabirds-data/nabirds'
image_path = os.path.join(dataset_path + '/images/')


# Set seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
_ = torch.manual_seed(SEED)


# Enable CUDA
if torch.cuda.is_available():
    device = 'cuda:0'
    print('CUDA enabled!')
else:
    device = 'cpu'
    print('CPU enabled.')


# Input image transformations
class CenterCrop(torch.nn.Module):
    def __init__(self, percent=0.8):
        super().__init__()
        self.frac = percent
    
    def forward(self, img):
        return v2.functional.center_crop(img, (int(self.frac*img.size[1]), int(self.frac*img.size[0])))

class rot90(torch.nn.Module):
    def __init__(self, percent=0.8):
        super().__init__()
        self.frac = percent
    
    def forward(self, img):
        return v2.functional.rotate(img, 90)

trans = v2.Compose([          #v2.RandomRotation(180),
                              #CenterCrop(),
                              #v2.RandomResizedCrop(224),
                              v2.Resize((384,384)),
                              rot90(),
                              #v2.RandomHorizontalFlip(),
                              v2.PILToTensor(),
                              v2.ConvertImageDtype(torch.float),
                              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

invTrans = v2.Compose([ v2.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                v2.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


# Load useful interfacing variables
import pickle
hierarchical_labels = {}
with open('./labels/hierarchical_class_labels.pkl', 'rb') as f:
    hierarchical_labels = pickle.load(f)

normalized_labels = []
with open('./labels/normalized_class_labels_fixed.pkl', 'rb') as f:
    normalized_labels = pickle.load(f)

print(normalized_labels)

normalized_class_names = []
with open('./labels/normalized_class_mappings.pkl', 'rb') as f:
    normalized_class_names = pickle.load(f)



# Load Dataset
full_dataset = torchvision.datasets.ImageFolder(image_path, transform=trans)
split = 0.5
length = 48562
batch_size = 8
train_size = int((1-split) * length)
test_size = length - train_size

ds_train_idx, ds_test_idx = train_test_split(list(range(len(full_dataset))), test_size=split, stratify=full_dataset.targets, random_state=SEED)
ds_test = torch.utils.data.Subset(full_dataset, ds_test_idx)



# Load Model
from Model.CascadeHead import CascadeHead

def load_model(mtype='CCC'):
    model = None
    
    if mtype == 'CCC':
        print('Loaded CCC Model')
        # May need to be changed
        model = torch.load('Saved_Models/CumulativeCascadeHead_Swin_b_384_Layer1_2_3_2025-04-28_04:20:06.348642.pt', weights_only=False).to(device)
    elif mtype == 'NCC':
        print('Loaded NCC Model')
        # May need to be changed
        model = torch.load('Saved_Models/CascadeHead_Swin_b_384_Layer1_2_3_2025-04-27_06:55:07.925976.pt', weights_only=False).to(device)
    elif mtype == 'LC':
        print('Loaded LC Model')
        # May need to be changed
        model = torch.load('Saved_Models/LinearHead_Swin_b_384_Layer1_2_3_2025-04-27_20:30:55.190130.pt', weights_only=False).to(device)
    else:
        sys.exit(0)
        
    for param in model.parameters():
        param.requires_grad = False
    model.zero_grad()
    model.eval()

    return model



# TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

def get_idx_from_test(dset=ds_test, idx=range(555)):
    desired_idx = []
    desired_labels = []
    c_idx = 0
    for img, label in dset:
        if (c_idx % 1000) == 0: print(c_idx) 
        if label in idx: 
            desired_idx.append(ds_test_idx[c_idx])
            desired_labels.append(label)
        c_idx += 1 
    return desired_idx, desired_labels

def get_cached_idx_from_test(idx=[]):
    want_idx = None
    with open('./labels/all_test_idx.pkl', 'rb') as f:
        want_idx = np.array(pickle.load(f))

    want_labels = None
    with open('./labels/all_test_class.pkl', 'rb') as f:
        want_labels = np.array(pickle.load(f))

    good = []
    if idx != []:
        for i in idx:
            good += np.where(want_labels == i)[0].tolist()

        want_idx = want_idx[np.array(good)]
        want_labels = want_labels[np.array(good)]
        
    return want_idx, want_labels


def generate_features(model, dataset, instance_count, output_shape=[batch_size, 12*12*1024]):
    model.head = torch.nn.Identity()
    features = torch.zeros(instance_count, output_shape[1])

    j = 0
    for i, (imags, _) in zip(tqdm.trange(instance_count // batch_size), dataset):
        with torch.no_grad():
            feature = torch.flatten(model(imags.to(device)), start_dim=1, end_dim=-1)
        features[j*batch_size:(j*batch_size + imags.shape[0])] = feature.detach().cpu()
        j += 1
   
    #pca = PCA(n_components=50)
    #features = pca.fit_transform(features.numpy())

    return features

def get_tsne(model, find_labels, save=False, name='embedded_tsne.pkl'):
    want_idx, want_labels = get_cached_idx_from_test(idx=find_labels)

    total_instances = len(want_idx)
    ds_tsne = torch.utils.data.Subset(full_dataset, want_idx)
    tsne_dataloader = torch.utils.data.DataLoader(ds_tsne, batch_size)

    features = generate_features(model, tsne_dataloader, total_instances)

    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    embedded_features = tsne.fit_transform(features)

    if save:
        with open('./labels/' + name, 'wb') as f:
            pickle.dump(embedded_features ,f)

    return embedded_features, want_idx, want_labels

# Adapted from https://github.com/mobulan/MPSA/blob/main/visualize/visualization.py, citation [12] in the paper!
def plot_tsne(embedded_features, want_idx, want_labels, from_file=False, fpath='', idclass=3, show=False, save=False, name='tsne.png', target=[]):
    lens = [2, 50, 404, 555]
    num_class = lens[idclass]
    difficult_class = find_labels = range(num_class)

    if from_file:
        want_idx, want_labels = get_cached_idx_from_test(target)
        with open(fpath, 'rb') as f:
            embedded_features = pickle.load(f)

        embedded_features = embedded_features[want_idx]

    print(want_labels)
    print(embedded_features.shape)

    total_instances = embedded_features.shape[0]
    want_labels = want_labels.tolist()
   

    colors = plt.cm.rainbow(np.linspace(0, 1, num_class))
    b = torch.randperm(num_class)
    colors = colors[b]
    marker_size = 12

    plt.figure(figsize=(10,8))

    for i in range(total_instances):
        if (i % 2000) == 0: print(i)
        plt.scatter(embedded_features[i, 0],
            embedded_features[i, 1],
            marker='*' if want_labels[i] == 264 else '.',
            color=colors[find_labels.index(normalized_labels[want_labels[i]][idclass])],
            label=str(want_labels[int(i)]), s=40 if want_labels[i] == 264 else marker_size)

    plt.legend().set_visible(False)
    plt.axis('off')
    if show: plt.show()
    if save: plt.savefig('./Output Images/' + name)

# Female hummingbird classes
fhbs = [256, 258, 260, 262, 264, 266, 268, 270, 272]
# Hummingbird classes
hbs = [255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272]
# TSNE Outputs for Cumulative Cascade Classifier (BEST MODEL)
#print('-'*25)
#print('Running CCC t-SNE')
model = load_model(mtype='CCC')
emb_feat, wanti, wantl = get_tsne(model, hbs, save=False)
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/tsne_embedded_test.pkl', idclass=0, show=False, save=True, name='t-SNE/CCC/CCC_TSNE_c1.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/tsne_embedded_test.pkl', idclass=1, show=False, save=True, name='t-SNE/CCC/CCC_TSNE_c2.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/tsne_embedded_test.pkl', idclass=2, show=False, save=True, name='t-SNE/CCC/CCC_TSNE_c3.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/tsne_embedded_test.pkl', idclass=3, show=False, save=True, name='t-SNE/CCC/CCC_TSNE_c4.png')

# Plot hummingbird classes
plot_tsne(emb_feat, wanti, wantl, from_file=False, fpath='./labels/tsne_embedded_test.pkl', idclass=3, show=False, save=True, name='t-SNE/CCC/CCC_TSNE_hummingbirds.png', target=hbs)


# TSNE Outputs for Noncumulative Cascade Classifier (2nd BEST MODEL)
print('-'*25)
print('Running NCC t-SNE')
#model = load_model(mtype='NCC')
#get_tsne(model, find_labels=[], save=True, name='NCC_embedded_tsne.pkl')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/NCC_embedded_tsne.pkl', idclass=0, show=False, save=True, name='t-SNE/NCC/NCC_TSNE_c1.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/NCC_embedded_tsne.pkl', idclass=1, show=False, save=True, name='t-SNE/NCC/NCC_TSNE_c2.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/NCC_embedded_tsne.pkl', idclass=2, show=False, save=True, name='t-SNE/NCC/NCC_TSNE_c3.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/NCC_embedded_tsne.pkl', idclass=3, show=False, save=True, name='t-SNE/NCC/NCC_TSNE_c4.png')


# TSNE Outputs for Linear Classifier (CONTROL MODEL)
print('-'*25)
print('Running LC t-SNE')
#model = load_model(mtype='LC')
#get_tsne(model, find_labels=[], save=True, name='LC_embedded_tsne.pkl')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/LC_embedded_tsne.pkl', idclass=0, show=False, save=True, name='t-SNE/LC/LC_TSNE_c1.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/LC_embedded_tsne.pkl', idclass=1, show=False, save=True, name='t-SNE/LC/LC_TSNE_c2.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/LC_embedded_tsne.pkl', idclass=2, show=False, save=True, name='t-SNE/LC/LC_TSNE_c3.png')
_ = torch.manual_seed(SEED)
#plot_tsne(None, None, None, from_file=True, fpath='./labels/LC_embedded_tsne.pkl', idclass=3, show=False, save=True, name='t-SNE/LC/LC_TSNE_c4.png')
