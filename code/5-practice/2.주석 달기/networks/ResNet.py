import torch
import torch.nn as nn
class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            #입력3 / 출력64 / kernel size=7 / stride=2 / padding=3(이미지 크기 유지 위해) 
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ),
        #maxpool => kernel size=3 / stride=2 / padding=1(이미지 크기 절반으로 만들기 위해) 
        self.pool=nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x=self.pool(x)
        x=self.fc(x)
        return x
    
class ResNet_back(nn.Module):
    def __init__(self,num_classes=10,config='18'):
        super().__init__()
        #AdaptiveAvgPool2d=7x7 입력을 강제적으로 1x1로 바꿔줌.
        self.pool=nn.AdaptiveAvgPool2d(1)
        #입력 configuration에 따라 average pooling에 입력되는 채널 크기가 달라짐. 
        in_feat=512 if config in ['18','34'] else 2048
        self.fc=nn.Linear(in_feat,num_classes)
    def forward(self,x):
        x=self.pool(x)
        x=self.fc(x)
        return x
    
class ResNet_back(nn.Module):
    def __init__(self,num_classes=10,config='18'):
        super().__init__()
        
    def forward(self,x):
        return x
