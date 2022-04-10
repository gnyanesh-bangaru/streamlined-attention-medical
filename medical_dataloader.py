# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:38:52 2022

@author: Lalith_B
"""

import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from glob import glob
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import errno
from PIL import Image
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from CUDA_data_switch import DeviceDataLoader, get_default_device
from torch.utils.data import Subset, DataLoader

from HAM10000 import HAM10000, data

cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING=1

class MedicalData:
    
    def __init__(self, batch_size:int):
        super(MedicalData, self).__init__()
        self.device = get_default_device()
        self.transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()
                                    ])
        self.batch_size = batch_size
        
    def train_val_dataset(self, dataset, val_split=0.20):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['test'] = Subset(dataset, val_idx)
        return datasets
            
    
    def tumor_binary(self):
        num_classes = 2
        dataset = ImageFolder(r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\Medical Datasets\Tumor Binary\brain_tumor_dataset', 
                              self.transforms)
        datasets = self.train_val_dataset(dataset)
        trainset = datasets['train']
        testset = datasets['test']
        
        train_dl = DataLoader(trainset, self.batch_size, shuffle=True) 
        test_dl = DataLoader(testset, self.batch_size, shuffle=True) 
        
        train_dataloader = DeviceDataLoader(train_dl, self.device)
        test_dataloader = DeviceDataLoader(test_dl, self.device)
        
        return train_dataloader, test_dataloader, num_classes
        
    
    def tumor_multi(self):
        num_classes = 4
        trainset = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\Medical Datasets\Tumor Multi-Class\Training'
        testset = r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\Medical Datasets\Tumor Multi-Class\Testing'
        
        train_dataset = ImageFolder(os.path.join(trainset), self.transforms)
        test_dataset = ImageFolder(os.path.join(testset), self.transforms)
        
        train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True) 
        test_dl = DataLoader(test_dataset, self.batch_size, shuffle=True)
        
        train_dataloader = DeviceDataLoader(train_dl, self.device)
        test_dataloader = DeviceDataLoader(test_dl, self.device)
        
        return train_dataloader, test_dataloader, num_classes
        
    
    def covid_binary(self):
        num_classes = 2
        dataset = ImageFolder(r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\Medical Datasets\COVID Binary', 
                              self.transforms)
        datasets = self.train_val_dataset(dataset)
        trainset = datasets['train']
        testset = datasets['test']
        
        train_dl = DataLoader(trainset, self.batch_size, shuffle=True) 
        test_dl = DataLoader(testset, self.batch_size, shuffle=True) 
        
        train_dataloader = DeviceDataLoader(train_dl, self.device)
        test_dataloader = DeviceDataLoader(test_dl, self.device)
        
        return train_dataloader, test_dataloader, num_classes
        
    
    def covid_multi(self):
        num_classes = 3
        dataset = ImageFolder(r'C:\Users\Lalith_B\Desktop\Gnyan ResNet\Datasets\Medical Datasets\COVID Multi-Class\original_images', 
                              self.transforms)
        datasets = self.train_val_dataset(dataset)
        trainset = datasets['train']
        testset = datasets['test']
        
        train_dl = DataLoader(trainset, self.batch_size, shuffle=True) 
        test_dl = DataLoader(testset, self.batch_size, shuffle=True) 
        
        train_dataloader = DeviceDataLoader(train_dl, self.device)
        test_dataloader = DeviceDataLoader(test_dl, self.device)
        
        return train_dataloader, test_dataloader, num_classes 
    
    def ham10000(self):
        num_classes = 7
        train, test = data()
        
        trainset = HAM10000(train, self.transforms) 
        testset = HAM10000(test, self.transforms)
        
        train_dl = DataLoader(trainset, self.batch_size, shuffle=True)
        test_dl = DataLoader(testset, self.batch_size, shuffle=True)
        
        train_dataloader = DeviceDataLoader(train_dl, self.device)
        test_dataloader = DeviceDataLoader(test_dl, self.device)
        
        return train_dataloader, test_dataloader, num_classes 
         
# a = MedicalData()
# a