import torch
import torch.nn as nn
class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            #입력3(rgb) / 출력64 / kernel size=7 / stride=2 / padding=3(이미지 크기 유지 위해) 
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
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
    
#3-3 구조-Block
class ResNet_Block(nn.Module):
    def __init__(self,in_channel,out_channel,downsampling=False):
        super().__init__()
        self.downsampling=downsampling
        #다운샘플=입력 데이터의 크기와 네트워크를 통과한 출력 데이터 크기가 다를 경우 맞춰주는 것. 이 경우에는 stride를 2로 해서 적용함(기본 stride값은 1)
        stride=1
        if self.downsampling:
            stride=2
            self.skip_conv=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,3,stride,1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )
        #다운샘플을 안 할 경우
        self.first_conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.second_conv=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.relu=nn.ReLU()
        
    def forward(self,x):
        #다운샘플할 경우 x를 복제하고 그걸 skip_conv에 태운다. 
        skip_x=torch.clone(x)
        if self.downsampling:
            skip_x=self.skip_conv(skip_x)
        x=self.first_conv(x)
        x=self.second_conv(x)
        #x와 identify x(?)를 합친다.
        x=x+skip_x
        x=self.relu(x)
        return x
    
class ResNet_middle(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 숫자, 출력 숫자, block 갯수, 다운샘플링.
        #64의 입력을 받고 64의 출력을 내보내는 block이 두 개.
        self.layer1=self.make_layer(64,64,2)
        #64의 입력을 받고 128의 출력을 내보내는 block이 두 개.
        self.layer2=self.make_layer(64,128,2,True)
        self.layer3=self.make_layer(128,256,2,True)
        self.layer4=self.make_layer(256,512,2,True)
        
    def make_layer(self,in_channel,out_channel,num_block,downsampling=False):
        layer=[ResNet_Block(in_channel,out_channel,downsampling)]
        for _ in range(num_block-1):
            layer.append(ResNet_Block(out_channel,out_channel))
        return nn.Sequential(*layer)
        #언패킹.리스트는 별 하나, 딕셔너리는 별 두 개.키와 벨류가 있으니까. 
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.front=ResNet_front()
        self.middle=ResNet_middle()
        self.back=ResNet_back()
        
    def forward(self,x):
        x=self.front(x)
        x=self.middle(x)
        x=self.back(x)
        return x
