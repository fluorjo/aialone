import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.optim import Adam

import cv2

from torch.utils.data import DataLoader


batch_size=100
hidden_size=500
num_classes=10
lr=0.001
epochs=3

img_size=32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#이미지 크기 변경&텐서로 변경
#여러 트랜스폼을 묶어서 하나로 구성, 함수의 형태로. = compose
train_dataset=CIFAR10(root='./cifar', train=True, download=True)
mean=train_dataset.data.mean(axis=(0,1,2))/255.0
std=train_dataset.data.std(axis=(0,1,2))/255.0

transform=Compose([
    Resize((img_size,img_size)),
    ToTensor(),
    Normalize(mean,std)
])

train_dataset=CIFAR10(root='./cifar', train=True, download=True,transform=transform,)
test_dataset=CIFAR10(root='./cifar', train=False, download=True,transform=transform,)

#dataloader
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)

#데이터셋 이미지 확인하기(텐서->넘파이)

def reverse_trans(x):
    x=(x*std)+mean
    return x.clamp(0,1)*255

def get_numpy_image(data):
    img=reverse_trans(data.permute(1,2,0)).type(torch.uint8).numpy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx=1000
img,label=train_dataset.__getitem__(idx)
img=cv2.resize(get_numpy_image(img),(512,512))
label=labels[label]

#모델 클래스,
class myMLP(nn.Module):
    def __init__(self, hidden_size,num_classes):
        super().__init__()
        self.fc1=nn.Linear(28*28,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        b,c,w,h=x.shape
        x=x.reshape(-1,28*28)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        return x

class myLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.bn1=nn.BatchNorm2d(num_features=6)