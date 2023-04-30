import torch
import torch.nn as nn

class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
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
        x=self.fc(x)
        return x
class ResNet_Block(nn.Module):
    def __init__(self,in_channel,out_channel,downsampling=False):
        super().__init__()
        self.downsampling=downsampling
        stride=1
        if self.downsampling:
            stride=2
            self.skip_conv=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,3,stride,1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        self.first_conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.second_conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.relu=nn.ReLU()
    def forward(self,x):
        skip_x=torch.clone(x)
        if self.downsampling:
            skip_x=self.skip_conv(skip_x)
        x=self.first_conv(x)
        x=self.second_conv(x)
        x=x+skip_x
        x=self.relu(x)
        return x

class ResNet_middle(nn.Module):
    def __init__(self):
        self.layer1=self.make_layer(64,64,2)
        self.layer2=self.make_layer(64,128,2,True)
        self.layer3=self.make_layer(128,256,2,True)
        self.layer4=self.make_layer(256,512,2,True)
    def make_layer(self,in_channel,out_channel,num_block, downsampling=False):
        layer=[ResNet_Block(in_channel,out_channel,downsampling)]
        for _ in range(num_block-1):
            layer.append(ResNet_Block(out_channel,out_channel))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x
class ResNet(nn.Module):
    def __init__(self) :
        super().__init__()
        self.front=ResNet_front()
        self.middle=ResNet_middle()
        self.back=ResNet_back()
    def forward(self,x):
        x=self.front(x)
        x=self.middle(x)
        x=self.back(x)
        return x