import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision.transforms import v2
from PIL import Image
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class nabirdsDataset(Dataset):
    def __init__(self, dataset_path, image_path, ignore=[], general=True, normalized_names=None):
        self.dataset_path = dataset_path
        self.normalized_names = normalized_names
        self.image_path = image_path
        self.image_paths = self.load_image_paths()
        self.image_sizes = self.load_image_sizes()
        self.image_bboxes = self.load_bounding_box_annotations()
        self.image_bboxes_np = self.image_bboxes.to_numpy()
        self.part_names = self.load_part_names()
        self.image_parts = self.load_part_annotations()
        self.image_class_labels = self.load_image_labels()
        self.class_names = self.load_class_names()
        self.general = general
        self.class_names_rectified, self.class_names_dict, self.image_class_labels_rectified, self.image_class_labels_dict = self.rectify_class_names()

        if ignore != []:
            self.image_paths = self.image_paths[self.image_paths.id.isin(ignore)].reset_index()
            self.image_sizes = self.image_sizes[self.image_sizes.id.isin(ignore)].reset_index()
            self.image_bboxes = self.image_bboxes[self.image_bboxes.id.isin(ignore)].reset_index()
            self.image_parts = self.image_parts[self.image_parts.id.isin(ignore)].reset_index()
            self.image_class_labels = self.image_class_labels[self.image_class_labels.id.isin(ignore)].reset_index()

        self.length = len(self.image_paths)
        self.classes = len(list(self.class_names_rectified['class_id'])) 
        self.general_classes = 22

        self.cdict, self.otndict = self.dictionary_classes()
    
    class rot90(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):
            return v2.functional.rotate(img, 90)

    def rectify_class_names(self):
        unique_ids = np.unique(self.image_class_labels['class_id'].to_numpy()).astype(str)
        
        rectified_class_names = pd.DataFrame(data={'class_id': unique_ids})
        rectified_class_names['rectified_id'] = list(rectified_class_names.index)

        class_names_dict = dict(zip(rectified_class_names['rectified_id'], rectified_class_names['class_id']))
        reverse_class_names_dict = dict(zip(rectified_class_names['class_id'], rectified_class_names['rectified_id']))

        image_class_labels_rectified = self.image_class_labels.copy()
        corr = []
        for id in list(image_class_labels_rectified['class_id'].to_numpy().astype(str)):
            corr.append(reverse_class_names_dict[id])

        image_class_labels_rectified['rectified_id'] = corr
        image_class_labels_dict = dict(zip(image_class_labels_rectified['rectified_id'], image_class_labels_rectified['class_id']))

        return rectified_class_names, class_names_dict, image_class_labels_rectified, image_class_labels_dict


    def load_image_paths(self):
        colnames = ['id', 'path']
        paths = pd.read_csv(self.dataset_path + 'images.txt', sep=' ', names=colnames, header=None)

        return paths

    
    def load_image_sizes(self):
        colnames = ['id', 'width', 'height']
        sizes = pd.read_csv(self.dataset_path + 'sizes.txt', sep=' ', names=colnames, header=None)

        return sizes


    def load_bounding_box_annotations(self):
        colnames = ['id', 'x1', 'y1', 'x2', 'y2']
        bboxes = pd.read_csv(self.dataset_path + 'bounding_boxes.txt', sep=' ', names=colnames, header=None)

        return bboxes

    
    def load_part_names(self):
        colnames = ['part_id', 'part_name']
        part_names = pd.read_csv(self.dataset_path + 'parts/part_locs.txt', sep=' ', names=colnames, header=None)

        return part_names


    def load_part_annotations(self):
        colnames = ['id', 'part_id', 'x', 'y', 'in_img']
        parts = pd.read_csv(self.dataset_path + 'parts/part_locs.txt', sep=' ', names=colnames, header=None)

        return parts

    
    def load_image_labels(self):
        colnames = ['id', 'class_id', 'general_class_id']
        labels = pd.read_csv(self.dataset_path + 'image_class_labels.txt', sep=' ', names=colnames, header=None)
        labels.sort_values('class_id', inplace=True, ignore_index=True)
        labels['general_class_id'] = pd.read_csv('./labels/general_class_labels.txt', header=None)

        return labels

    
    def load_class_names(self):
        colnames = ['class_id', 'class']
        names = np.array(colnames)
  
        with open(os.path.join(self.dataset_path, 'classes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = pieces[0]
                names = np.vstack([names, [class_id, ' '.join(pieces[1:])]])

        classes = pd.DataFrame(names[1:,:], columns=colnames)

        return classes

    def dictionary_classes(self):
        cdict = {}
        for i in self.image_class_labels.index:
            id = self.image_class_labels['id'].values[i]
            cid = self.image_class_labels['class_id'].values[i]
            
            cdict[id] = cid

        oldcid_to_newcid = {}
        for i in self.class_names.index:
            oldcid = self.class_names['class_id'].values[i]
            name = self.class_names['class'].values[i]
            try:
                newcid = self.normalized_names[3].index(name)
                oldcid_to_newcid[int(oldcid)] = newcid
            except: continue
          
        return cdict, oldcid_to_newcid

    
    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.image_path + self.image_paths['path'].values[idx]
        id = self.image_paths['id'].values[idx]
        label = self.otndict[self.cdict[id]]

        tf = v2.Compose([      #v2.RandomRotation(180),
                              lambda x: Image.open(x).convert('RGB'),
                              #CenterCrop(),
                              #v2.RandomResizedCrop(224),
                              v2.Resize((384,384)),
                              self.rot90(),  
                              #v2.RandomHorizontalFlip(),
                              v2.PILToTensor(),
                              v2.ConvertImageDtype(torch.float),
                              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        img = tf(img_path)
        #plt.imshow(img)
        #plt.show()

        return img, label


# def main():
#     # Paths
#     dataset_path = 'nabirds-data/nabirds/'
#     image_path  = dataset_path + 'images'
    
    # Create dataset
    #data = nabirdsDataset(dataset_path, image_path)

if __name__ == '__main__':
    # Paths
    dataset_path = 'nabirds-data/nabirds/'
    image_path  = dataset_path + 'images/'

    #index = np.arange(555)
    #print(index)

    # Create dataset
    data = nabirdsDataset(dataset_path, image_path)
    unique_ids = np.unique(data.image_class_labels['class_id'].to_numpy()).astype(str)

    img, label = data.__getitem__(441)

    datalodea = DataLoader(data, batch_size=128, shuffle=False, num_workers=4)

    

    #print(data.class_names.loc[data.class_names['class_id'].isin(list(unique_ids))])
