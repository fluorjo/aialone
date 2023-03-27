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
transform = Compose({
    Resize((img_size,img_size)),
    ToTensor(),
    Normalize(mean,std)
})

train_dataset=CIFAR10(root='./cifar', train=True, transform=transform, download=True)
test_dataset=CIFAR10(root='./cifar', train=True, transform=transform, download=True)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

class myMLP(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1=nn.Linear(28*28,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
    def forward(self, x):
        b,w,h,c = x.shape
        x=x.reshape(-1,28*28)
        
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        
        return x
class myLeNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        self.bn1 =nn.BatchNorm2d(num_features=6)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        
        self.bn2 =nn.BatchNorm2d(num_features=16)
        self.act2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
       
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
class myLeNet_seq(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
       
        self.seq1=nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        nn.BatchNorm2d(num_features=6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
        
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        ),
        self.seq2=nn.Sequential(
            nn.Linear(in_features=16*5*5,out_features=120),
            nn.Linear(in_features=120,out_features=84),
            nn.Linear(in_features=84,out_features=num_classes),
        ),
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
class myLeNet_Linear(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        self.bn1 =nn.BatchNorm2d(num_features=6)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        
        self.fc_1=nn.Linear(6*14*14,2048)
        self.fc_2=nn.Linear(2048,6*14*14)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        
        self.bn2 =nn.BatchNorm2d(num_features=16)
        self.act2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
       
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        
        _,tmp_c,tmp_w,tmp_h=
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

class myLeNet_convs(nn.Module):
    def __init__(self,N, num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        self.bn1 =nn.BatchNorm2d(num_features=6)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        
        self.fc_1=nn.Linear(6*14*14,2048)
        self.fc_2=nn.Linear(2048,6*14*14)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        
        self.bn2 =nn.BatchNorm2d(num_features=16)
        self.act2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
       
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        for module in self.tmp_conv1
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        
        _,tmp_c,tmp_w,tmp_h=
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

class myLeNet_inception(nn.Module):
    def __init__(self,N, num_classes):
        super().__init__()
        self.conv1_1=nn.Conv2d(3,6,5,1,2)
        self.conv1_2=nn.Conv2d(3,6,3,1,1)
        self.conv1_3=nn.Conv2d(3,6,1,1,0)
        self.conv1=nn.Conv2d(in_channels=18,out_channels=6,kernel_size=5)
        #배치 노말라이즈 = 값이 큰 애들, 작은 애들끼리 각각 노말라이징해서 분포를 적당하게 맞춤.
        #몇 개의 채널을 넣을 건지 명시해줘야.
        self.bn1 =nn.BatchNorm2d(num_features=6)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        
        self.fc_1=nn.Linear(6*14*14,2048)
        self.fc_2=nn.Linear(2048,6*14*14)
        
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        
        self.bn2 =nn.BatchNorm2d(num_features=16)
        self.act2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
       
    #forward에서는 데이터를 하나 받게 돼있음. 그게 x. 일종의 약속임.
    #만약 x가 튜플이면 x=(a,b)같은 식으로 강제로 나눠줌.
    def forward(self, x):
        b,c,h,w = x.shape
        x_1=self.conv1_1(x)
        x_2=self.conv1_2(x)
        x_3=self.conv1_3(x)
        
        x_cat=torch.cat((x_1,x_2,x_3),dim=1)
        
        
        x=self.conv1(x_cat)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.pool1(x)
        
        _,tmp_c,tmp_w,tmp_h=
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

#model=myMLP(hidden_size,num_classes).to(device)
model=myLeNet_inception(num_classes).to(device)
loss=nn.CrossEntropyLoss()
optim=Adam(model.parameters(),lr=lr)

def eval(model, loader):
    total=0
    correct=0
    for idx, (image, target) in enumerate(train_loader):
        image=image.to(device)
        target=target.to(device)
        
        out=model(image)
        #'값',값의 '위치' - 두 개가 나옴. 그래서 len 하면 2.
        #몇점으로/누가 1등을 했는지가 나오는 것. 근데 몇점인지는 안 궁금. '누가'가 중요.
        
        _,pred=torch.max(out,1)
        #같으면 true(1) 틀리면 false(0). 그걸 다 더함.
        correct+= (pred ==target).sum().item()
        total+=image.shape[0]
    return correct/total
def eval_class(model, loader):
    total=torch.zeros(num_classes)
    correct=torch.zeros(num_classes)
    for idx, (image, target) in enumerate(train_loader):
        image=image.to(device)
        target=target.to(device)
        
        out=model(image)
        #'값',값의 '위치' - 두 개가 나옴. 그래서 len 하면 2.
        #몇점으로/누가 1등을 했는지가 나오는 것. 근데 몇점인지는 안 궁금. '누가'가 중요.
        
        _,pred=torch.max(out,1)
        #같으면 true(1) 틀리면 false(0). 그걸 다 더함.
        for i in range(num_classes):
            #0에는 0을 0이라고 한 것만 들어가야 한다. 각각 트루일 때 and연산(=곱연산) 
            #correct[i]+= (target==i)&(pred==i).sum().item()
            correct[i]+= (target==i)*(pred==i).sum().item()
            total[i]+=(target==i).sum().item()
    return correct,total

for epoch in range(epochs):
    for idx, (image, target) in enumerate(train_loader):
        image=image.to(device)
        target=target.to(device)
        
        out=model(image)
        loss_value=loss(out,target)
        
        optim.zero_grad()
        
        loss_value.backward()
        optim.step()
        
        if idx%100 ==0:
            print(loss_value.item())
            print('accuracy:',eval(model,test_loader))
            


        