

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import glob 
from torch.utils.data import Subset
import torch




def train(model,training_loader,num_epochs = 5, lr = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    training_count = len(training_loader)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_function =torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for _ , (input,known,target) in enumerate(training_loader):

            input = input.float().to(device)
            target = torch.flatten(target).to(device)
            known = known.to(device)
            
            outputs = model(input)
            predicted = outputs[known<1].to(device)
            target = target.float()
            predicted = predicted.float()

            loss = loss_function(predicted, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        training_loss = train_loss/training_count
        
        print(f'Epoch={epoch+1}/{num_epochs},   Loss={training_loss}')
    
    return model

def evaluate(model,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_count = len(test_loader)
    loss_function =torch.nn.MSELoss()
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        test_loss = 0.0
        for _ , (input,known,target) in enumerate(test_loader):

            input = input.float().to(device)
            target = torch.flatten(target).to(device)
            known = known.to(device)
            
            outputs = model(input)
            predicted = outputs[known<1].to(device)
            target = target.float()
            predicted = predicted.float()

            loss = loss_function(predicted, target)
            
            test_loss += loss.item()

        test_loss = test_loss/test_count
        
    print(f'Evaluation Loss= {test_loss}')