import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import glob 
from torch.utils.data import Subset
import torch
import torch.nn as nn
from torch import nn 
from image_alteration import image_alteration
from PIL import Image
import pickle
from matplotlib import pyplot as plt
import random


class ImageData(Dataset):
    def __init__(self,im_list,original = False) -> None:
        self.im_list = im_list
        self.original = original

    def __getitem__(self, idx):
        im = Image.open(self.im_list[idx]).convert('RGB')
        resized_im = transforms.Resize((100,100))(im)
        if self.original:
            return np.asarray(resized_im) 
        else:
            offset_1,offset_2 = random.randint(0, 16),random.randint(0, 16)
            spacing_1,spacing_2 = random.randint(2,6),random.randint(2,6)
            input_array,known_array,targeted_array = image_alteration(np.asarray(resized_im),[offset_1,offset_2],[spacing_1,spacing_2])
            input_array = torch.from_numpy(input_array/255)
            targeted_array = targeted_array/255
            return input_array,known_array,targeted_array
    
    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset
        
        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        n_samples = len(self.im_list)
        return n_samples