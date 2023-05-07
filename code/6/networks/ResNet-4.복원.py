import torch
import torch.nn as nn
#레이어 숫자에 따른 블록 개수를 리스트로 만들어놓기.
NUM_BLOCKS_18 = [2, 2, 2, 2]
NUM_BLOCKS_34 = [3, 4, 6, 3]
NUM_BLOCKS_50 = [3, 4, 6, 3]
NUM_BLOCKS_101 = [3, 4, 23, 3]
NUM_BLOCKS_152 = [3, 8, 36, 3]
#블록이 3,3일 때와 1,3,1일 때의 채널 수를 리스트로 만들어놓기.
NUM_CHANNEL_33 = [64, 64, 128, 256, 512]
NUM_CHANNEL_131 = [64, 256, 512, 1024, 2048]

class ResNet_front(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            #입력채널(3=rgb)/출력/커널사이즈/스트라이드/패딩
            nn.Conv2d(3,64,7,2,3),#이렇게 하면 이미지 크기 유지됨.
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ),
        #출력/커널사이즈/스트라이드/패딩 - 크기 절반으로 줄어들게.
        self.pool=nn.MaxPool2d(3,2,1),
    
    def forward(self,x):
        x=self.conv(x)
        x=self.pool(x)
        return x


class ResNet_back(nn.Module):
    def __init__(self,num_classes=10,config='18'):
        super().__init__()
        #AdaptiveAvgPool2d=7x7 입력을 강제적으로 1x1로 바꿔줌.
        self.pool=nn.AdaptiveAvgPool2d(1)
        #입력 configuration(레이어 수)에 따라 average pooling에 입력되는 채널 크기가 달라짐. 
        #18, 34(3*3구조)면 512, 나머지면 2048.
        in_feat=512 if config in ['18','34'] else 2048
        self.fc=nn.Linear(in_feat,num_classes)
    
    def forward(self,x):
        x=self.pool(x)
        #1*1*512 상태는 벡터가 아니다. shape를 맞춰서 마지막 1로 들어가게 해야 함. 
        #그래서 1인 차원들을 강제로 지워줌. 
        x=torch.squeeze(x)  
        x=self.fc(x)
        return x
    
    
class ResNet_Block(nn.Module):
    def __init__(self,in_channel,out_channel,downsampling=False):
        super().__init__()
        #다운샘플=입력 데이터의 크기와 네트워크를 통과한 출력 데이터 크기가 다를 경우 맞춰주는 것. 이 경우에는 stride를 2로 해서 적용함(기본 stride값은 1)
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
                nn.Conv2d(out_channel,out_channel,3,1,1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
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


class ResNet_BottleNeck(nn.Module):
    def __init__(self,in_channel,out_channel,downsampling=False):
        super().__init__()
        #다운샘플링할 경우 stride=2.
        #3*3에서 크기를 반으로 줄여야 하기 때문.
        stride=2 if downsampling else 1
        self.skip_conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        #1,3,1 구조에서 1x1들은 패딩이 없음.
        #이 구조에선 앞의 1,3은 마지막 아웃 채널의 4분의 1 만큼의 인풋이 들어감. 
        self.first_conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel//4,1,1,0),
            nn.BatchNorm2d(out_channel//4),
            nn.ReLU()
        )
        
        #1,3,1 구조에서 3x3은 패딩이 1. 
        self.second_conv=nn.Sequential(
                nn.Conv2d(out_channel//4,out_channel//4,3,stride,1),
                nn.BatchNorm2d(out_channel//4),
                nn.ReLU()
            )
        self.third_conv=nn.Sequential(
                nn.Conv2d(out_channel//4,out_channel,1,1,0),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        self.relu=nn.ReLU()
    
    def forward(self,x):
        skip_x=torch.clone(x)
        skip_x=self.skip_conv(skip_x)
        x=self.first_conv(x)
        x=self.second_conv(x)
        x=self.third_conv(x)
        x=x+skip_x
        x=self.relu(x)
        return x
    
class ResNet_middle(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config =='18':
            num_blocks,num_channel=NUM_BLOCKS_18,NUM_CHANNEL_33
            self.target_layer=ResNet_Block
        elif config =='34':
            num_blocks,num_channel=NUM_BLOCKS_34,NUM_CHANNEL_33
            self.target_layer=ResNet_Block
        elif config =='50':
            num_blocks,num_channel=NUM_BLOCKS_50,NUM_CHANNEL_131
            self.target_layer=ResNet_Block
        elif config =='101':
            num_blocks,num_channel=NUM_BLOCKS_101,NUM_CHANNEL_131
            self.target_layer=ResNet_Block
        elif config =='152':
            num_blocks,num_channel=NUM_BLOCKS_152,NUM_CHANNEL_131
            self.target_layer=ResNet_Block
        self.layer1=self.make_layer(num_channel[0],num_channel[1],num_blocks[0])
        self.layer2=self.make_layer(num_channel[1],num_channel[2],num_blocks[1])
        self.layer3=self.make_layer(num_channel[2],num_channel[3],num_blocks[2])
        self.layer4=self.make_layer(num_channel[3],num_channel[4],num_blocks[3])
    
    def make_layer(self,in_channel,out_channel,num_block,downsampling=False):
        #첫 번째 레이어에서 다운샘플링.
        layer=[self.target_layer(in_channel,out_channel,downsampling)]
        #나머지 레이어를 블록 수 -1 만큼 생성. 여기는 다운샘플링 안함.
        for _ in range(num_block-1):
            layer.append(self.target_layer(out_channel,out_channel))
        return nn.Sequential(*layer)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

class ResNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.front=ResNet_front()
        self.middle=ResNet_middle(args.res_config)
        self.back=ResNet_back(args.num_classes,args.res_config)
        
    def forward(self,x):
        x=self.front(x)
        x=self.middle(x)
        x=self.back(x)
        return x