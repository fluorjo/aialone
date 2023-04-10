import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import CIFAR10

#이미지 크기 변경&텐서로 변경, 정규화까지
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

#여러 트랜스폼을 묶어서 하나로 구성, 함수의 형태로 만들어주는 모듈
from torchvision.transforms import Compose

#영상 처리 라이브러리 - opencv 
import cv2
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#하이퍼 파라미터
batch_size=100
hidden_size=500
num_classes=10
lr=0.001
epochs=3

#이미지 크기 지정. 32
image_size=32
#트레인 데이터셋 넣고 평균, 표준편차 구해주기. ★트레인 데이터셋 '자체'의 평균, 표준편차 사용.
train_dataset=CIFAR10(root='./data',train=True,download=True)
mean=train_dataset.data.mean(axis=(0,1,2))/255.0
std=train_dataset.data.std(axis=(0,1,2))/255.0

#트랜스폼들을 하나의 함수로 만듦. 
transform = Compose([
    Resize((image_size,image_size)),
    ToTensor(),
    Normalize(mean,std)
])

#트레인, 데이터셋을 넣고 트랜스폼까지.
train_dataset=CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset=CIFAR10(root='./data',train=False,download=True,transform=transform)

#dataloader
train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

#데이터셋 이미지 확인하기(텐서->넘파이)
def reverse(x):
  x=(x*std)+mean
    #clamp = 입력으로 들어오는 값들을 (최소-최대) 범위 안으로 조정함. 
  return x.clamp(0,1)*255

def numpy_img(data):
    #permute=차원들의 순서를 재배치.
    img=reverse(data.permute(1,2,0)).type(torch.uint8).numpy()
    #BRG순서로 돼있는 걸 RGB로 바꿔야 함. 
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#라벨=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#불러올 인덱스=1000
idx=1000

#데이터를 가져오는데, loader에서 가져오고 transform 활용하는 방법으로.
img,label=train_dataset.__getitem__(idx)
#이미지 크기 조절. 512
img=cv2.resize(numpy_img(img),(512,512))
label=labels[label]
#,------------------lenet-----------------------#
class LeNet523(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #conv, 배치정규화, 렐루, maxpool - 2번.
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.bn1=nn.BatchNorm2d(num_features=6)
        self.relu1=nn.ReLU()
        self.pool1= nn.MaxPool2d(kernel_size=2)

        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.bn2=nn.BatchNorm2d(num_features=16)
        self.relu2=nn.ReLU()
        self.pool2= nn.MaxPool2d(kernel_size=2)
        
        #fully connected layer
        self.fc1=nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=num_classes)
    
    def forward(self,x):
        b,_,_,_=x.shape
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        x=x.reshape(b,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
#모델 선택

model=LeNet523(num_classes).to(device)

#loss
loss=nn.CrossEntropyLoss()
#optim
optim=Adam(model.parameters(),lr=lr)

#모델 정확도 계산 함수
def eval(model, loader):
    #전체와 맞춘 것을 일단 0으로.
  total=0
  correct=0
      #데이터 가져오고 디바이스에 넣기.
  for idx, (image, target) in enumerate(loader):
    image=image.to(device)
    target=target.to(device)
        #예측 값을 저장.
    out=model(image)
        #가장 높은 값을 가진 걸 다른 변수에 저장.
    _, pred=torch.max(out,1)
        #정확하게 맞춘 것과 전체를 각각 누적시킴.
    total+=image.shape[0]
    correct+=(pred==target).sum().item()
    #전체 중 맞춘 것 비율을 리턴.
    if idx==499:
      print('499total',total)
  return correct/total

#클래스 별 정확도 계산 함수. 
def eval_class(model,loader):
  #위에서 0으로 미리 둔 것처럼 0행렬을 만듦.
  total=torch.zeros(num_classes)
  correct=torch.zeros(num_classes)
    #데이터 가져오고 디바이스에 넣기.
  for idx,(image,target) in enumerate(loader):
    image=image.to(device)
    target=target.to(device)
        #예측 값을 저장.
    out=model(image)
        #가장 높은 값을 가진 걸 다른 변수에 저장.
    _,pred=torch.max(out,1)
        #클래스 별로 정확하게 맞춘 것과 전체 아이템 수를 각각 누적시킴.
  for i in range(num_classes):
      correct[i]+=((target==i)&(pred==i)).sum().item()
      total[i]+=(target==i).sum().item() 

#학습
for epoch in range(epochs):
    #enumerate로 인덱스와 데이터 가져오기
  for idx, (image, target) in enumerate(train_dataloader):
    image=image.to(device)
    target=target.to(device)
        #이미지를 모델에 넣어 출력값 도출.
    out=model(image)  
        #손실함수에 넣어 손실값 계산.
    loss_value=loss(out,target)
        #기울기 초기화
    optim.zero_grad()
        #역전파
    loss_value.backward()
        #파라미터 업데이트
    optim.step()
    if idx ==23:
      print('23')
      print('정확도', eval(model, test_dataloader))
    if idx ==499:
      print('stop')
      print('정확도', eval(model, test_dataloader))
    #  print('stop')
     # print(loss_value.item())
 