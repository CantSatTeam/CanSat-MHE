import os
import numpy as np
from skimage import io

import torchvision
from torch.utils.data import Dataset,DataLoader

from utils import (Nptranspose,Rotation,H_Mirror,V_Mirror)
# from utils import (RandomCrop,StdCrop)

class TrainDataset(Dataset):
    def __init__(self,image_dir,label_dir,transform=None):

        self.label_dir = label_dir
        self.image_dir = image_dir 
        
        self.data = []
        self.transform = transform

        files = os.listdir(self.label_dir)
        for item in files:
            if item.endswith(".png"):
                self.data.append(item.split(".png")[0][3:])
        self.data.sort()

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
 
        image = self.image_dir + "image" + self.data[index] + ".png"
        label = self.label_dir + "dsm" + self.data[index]+".png"

        image = io.imread(image)
        # print("image",image.shape)

        # image = np.reshape(image,(image.shape[0],image.shape[1],1))

        ### Replaced to work with grayscale label PNGs ###
        # label = io.imread(label) 
        # label = np.reshape(label,(label.shape[0],label.shape[1],1))

        # image = image.astype(np.float32)
        # label = label.astype(np.float32)
        ### End old code ###

        ### Start new code ###
        label = io.imread(label)
        
        # Handle multi-channel heightmaps (convert RGB to single channel)
        if len(label.shape) == 3:
            # If RGB, take just the first channel (they're all equal for grayscale)
            label = label[:, :, 0]
        
        # Ensure label is 3D (H, W, 1) for consistency
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=2)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        ### End new code ###

        # print(image)
    
        image = image/255.0
        # image = image.clip(min=0,max=1)
        label=label

        sample = {}
        sample["image"] = image
        sample["label"] = label
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


