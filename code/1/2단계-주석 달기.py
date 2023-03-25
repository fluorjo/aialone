#패키지 추가

#1.메인 네임스페이스(=변수, 함수 이름과 같은 명칭을 사용하는 공간으로 '소속'을 나타냄)
import torch
#2.
# nn.Module을 상속받는 다양한레이어와 모델 객체를 만듦
# 파이토치에서 인공 신경망 모델을 구성하는 핵심 구성 요소
# 우리가 만드는 custom model은 모두 이 클래스를 상속
# 신경망 모델을 구성하는 다양한 레이어, 함수를 제공.
import torch.nn as nn
#3.데이터를 텐서로 바꿔주는 패키지.
from torchvision.transforms import ToTensor

#4. 학습하고자 하는 모델의 파라미터값을 업데이트시키는 주체
from torch.optim import Adam


#.데이터셋 -mnist
from torchvision.datasets import MNIST
#사서. 데이터셋에서 가져온 데이터를 배치 사이즈만큼 모이면 네트워크로 데이터 넘겨줌. 데이터셋에 대한 파이썬 iterable(반복가능한) 객체 생성 클래스. 데이터셋을 미니배치 단위로 나누어주거나 순서를 섞기도 한다.
from torch.utils.data import DataLoader

#하이퍼 파라미터 설정
batch_size=100
hidden_size=500
num_classes=10
lr=0.001
epochs=3

#디바이스 설정. cuda 사용 가능하면 cuda, 아니면 cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#트레인/테스트 데이터셋 설정. 트랜스폼은 totensor
train_dataset=MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)
test_dataset=MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)

#트레인/테스트 데이터로더 설정. 셔플하기.
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#mlp 클래스 정의
class myMLP(nn.Module):
    #객체 생성. linear 4개.
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1=nn.Linear(28*28,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,hidden_size)
    #데이터 크기 변환
    def forward(self, x):
        b,w,h,c = x.shape
        x=x.reshape(-1,28*28)
        
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
    # 출력.
        return x

#mlp 클래스의 모델을 device로 전송 및 실행시킴. 
model=myMLP(hidden_size,num_classes).to(device)
#로스 함수. crossentropy
loss=nn.CrossEntropyLoss()
#옵티마이저.
optim=Adam(model.parameters(),lr=lr)

#학습시키기.
#epochs 만큼 반복시키기.
for epoch in range(epochs):
    #enumerate로 인덱스와 (이미지 데이터, 타겟(=정답))을 가져옴. 이게 데이터의 형태임.
    for idx, (image, target) in enumerate(train_loader):
        #이미지와 타겟 디바이스로 전송
        image=image.to(device)
        target=target.to(device)
        
        #image를 모델에 넣어 출력값 만들기.
        out=model(image)
        
        #출력값과 타겟을 손실함수에 넣어 손실값 계산
        loss_value=loss(out,target)
        
        #남아있을 수 있는 기울기를 초기화.
        optim.zero_grad()
        
        #기울기 계산. 역전파.
        loss_value.backward()
        
        #파라미터 업데이트.
        optim.step()
        
        if idx%100 ==0:
            print(loss_value.item())

        