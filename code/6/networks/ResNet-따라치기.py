import torch
import torch.nn as nn

class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ),
        self.pool=nn.MaxPool2d(3,2,1)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.pool(x)
        return x

class ResNet_back(nn.Module):
    def __init__(self,num_classes=10,config='18'):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        in_feat=512 if config in ['18','34'] else 2048
        self.fc=nn.Linear(in_feat,num_classes)
    
    def forward(self,x):
        x=self.pool(x)
        x=torch.squeeze(x)  
        x=self.fc(x)
        return x
    
class ResNet_Block(nn.Module):
    def __init__(self):
        super().__init__()
        
        

    