import torch
import torch.nn as nn
#입력부
class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool=nn.MaxPool2d(3,2,1)