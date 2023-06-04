


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
import cv2
import scipy.ndimage as ndimage

def visualize(model,image_arr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = model
    loaded_model.to(device)
    loaded_model.eval()
    
    original = Image.fromarray((image_arr.astype(np.uint8)))
    i,_,_ = image_alteration(image_arr,[2,2],[4,4])
    original_missing_pixels = Image.fromarray((i.transpose(2,1,0).astype(np.uint8)))
    original_missing_pixels_arr = i.transpose(2,1,0).astype(np.uint8)

    input = np.expand_dims(i, axis=0)
    input = torch.from_numpy(input).float().to(device)/255
    input = nn.Parameter(data=input, requires_grad=True)

    out = loaded_model(input)
    out = out.detach().cpu().numpy()
    out = out[0].transpose(2,1,0)*255
    model_prediction = out
    model_prediction[model_prediction<0] = original_missing_pixels_arr[model_prediction<0]
    model_prediction = Image.fromarray((model_prediction.astype(np.uint8)))

    angle = -90 # in degrees
    original_missing_pixels = ndimage.rotate(original_missing_pixels, angle, reshape=True)
    original_missing_pixels= cv2.flip(original_missing_pixels, 1)
    model_prediction= ndimage.rotate(model_prediction, angle, reshape=True)
    model_prediction= cv2.flip(model_prediction, 1)
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(original)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Original image")
    axes[1].imshow(original_missing_pixels)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Model input image")
    axes[2].imshow(model_prediction)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Predicted image")
    fig.tight_layout()