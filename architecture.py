import torch.nn as nn
import torch

class CNN(torch.nn.Module):
    def __init__(self,kernel_size = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 32, kernel_size= kernel_size, stride= 1, padding = "same")
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=kernel_size, stride= 1, padding = "same")
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size= kernel_size, stride= 1, padding = "same")
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels = 256, kernel_size= kernel_size, stride= 1, padding = "same")
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels = 128, kernel_size= kernel_size, stride= 1, padding = "same")
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=128, out_channels = 64, kernel_size= kernel_size ,stride= 1, padding = "same")
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels = 32, kernel_size= kernel_size, stride= 1, padding = "same")
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=32, out_channels = 3, kernel_size= kernel_size,stride= 1, padding = 'same')
        self.bn8 = nn.BatchNorm2d(3)

    def forward(self, input):

        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)

        output = self.conv6(output)
        output = self.bn6(output)
        output = self.relu6(output)

        output = self.conv7(output)
        output = self.bn7(output)
        output = self.relu7(output)
        
        output = self.conv8(output)
        output = self.bn8(output)

        return output