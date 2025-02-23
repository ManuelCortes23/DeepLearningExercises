from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import glob

class CustomDataset(Dataset):
    def __init__(self, path, n_classes=10, transform = False):
        
        self.transform = transform
        self.filelist = glob.glob(path+'/*.png')
        labels = np.zeros(len(self.filelist))
        for class_i in range(10):
            labels[ np.array(['class'+str(class_i) in x for x in self.filelist]) ] = class_i
            
        self.labels = torch.LongTensor(labels)  #... load the labels (copy from the notebook)
        
    def __len__(self):
       
        return len(self.filelist)

    def __getitem__(self, idx):
        
        img = Image.open(self.filelist[idx])
        
        x = transforms.ToTensor()( img ).view(-1) #.... transform to tensor and flatten it to a tensor of 69*69 = 4761
        
        y = self.labels[idx]
    
        return x, y