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


class cubDataset(Dataset):
    def __init__(self, dataset_path, image_path, train=True):
        self.dataset_path = dataset_path
        self.image_path = image_path
        self.train = train
        self.train_ids, self.test_ids = self.load_image_ids()
        
        if self.train:
            self.ids = self.train_ids
        else:
            self.ids = self.test_ids
        
        self.image_files = self.load_image_files()
        self.class_names = self.load_class_names()
        self.image_class_labels = self.load_image_class_labels()
        self.bounding_boxes = self.load_bounding_boxes()
        self.part_names = self.load_part_names()
        self.part_locs = self.load_part_locations()

        self.length = len(self.ids)

    
    class rot90(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):
            return v2.functional.rotate(img, 90)


    def load_image_ids(self):
        train_images = []
        test_images = []

        with open(os.path.join(self.dataset_path, 'train_test_split.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = int(pieces[0])
                is_train = int(pieces[1])
                if is_train:
                    train_images.append(image_id)
                else:
                    test_images.append(image_id)

        return train_images, test_images


    def load_image_files(self):
        colnames = ['id', 'filename']
        files = pd.read_csv(self.dataset_path + 'images.txt', sep=' ', names=colnames, header=None)
       
        files = files[files['id'].isin(self.ids)]

        return files


    def load_class_names(self):
        colnames = ['class_id', 'class_name']
        class_names = pd.read_csv(self.dataset_path + 'classes.txt', sep=' ', names=colnames, header=None)
        
        return class_names 
      
    
    def load_image_class_labels(self):
        colnames = ['id', 'class_id']
        image_class_labels = pd.read_csv(self.dataset_path + 'image_class_labels.txt', sep=' ', names=colnames, header=None)
        image_class_labels = image_class_labels[image_class_labels['id'].isin(self.ids)]
        
        return image_class_labels


    def load_bounding_boxes(self):
        colnames = ['id', 'x', 'y', 'width', 'height']
        bboxes = pd.read_csv(self.dataset_path + 'bounding_boxes.txt', sep=' ', names=colnames, header=None)
        
        bboxes = bboxes[bboxes['id'].isin(self.ids)]

        return bboxes


    def load_part_names(self):
        colnames = ['part_id', 'part_name']
        part_names = pd.read_csv(self.dataset_path + 'parts/parts.txt', sep=' ', names=colnames, header=None)

        return part_names


    def load_part_locations(self):
        colnames = ['id', 'part_id', 'x', 'y', 'visible']
        part_locs = pd.read_csv(self.dataset_path + 'parts/part_locs.txt', sep=' ', names=colnames, header=None)
        
        part_locs = part_locs[part_locs['id'].isin(self.ids)]

        return part_locs


    def load_part_click_locations(self):
        colnames = ['id', 'part_id', 'x', 'y', 'visible', 'time']
        part_click_locs = pd.read_csv(self.dataset_path + 'parts/part_click_locs.txt', sep=' ', names=colnames, header=None)
        
        part_click_locs = part_click_locs[part_click_locs['id'].isin(self.ids)]
    
        return part_click_locs


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.ids[idx]
        img_class_label = self.image_class_labels['class_id'].values[idx]
        img_name = self.image_files['filename'].values[idx]

        img_path = self.image_path + img_name

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

        return img, img_class_label-1


if __name__ == '__main__':
    dataset_path = 'CUB-200-2011/CUB_200_2011/'
    image_path = dataset_path + 'images/'
   
    test = cubDataset(dataset_path, image_path)

    img, label = test.__getitem__(0)

    print(img.shape)
    print(label)
