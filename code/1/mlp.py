#따라친 거, 주석 단 거, 주석부터 시작해서 구현해본 것 

#패키지 불러오기
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam
#클래스라서 객체로 만들기 위해 ()써줌.




from torch.utils.data import DataLoader

#하이퍼 파라미터 설정
batch_size=100
hidden_size=500
num_classes=10
lr=0.001
epochs=3
#cuda = gpu 랑 상호작용하게 하는 것. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#데이터 불러오기 
#dataset
train_dataset=MNIST(root='./mnist',train=True, transform=ToTensor(),download=True)
test_dataset=MNIST(root='./mnist',train=True, transform=ToTensor(),download=True)
#dataloader
train_loader= DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
test_loader=DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)
#전처리

#모델 클래스
class myMLP(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1=nn.Linear(28*28,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        b,w,h,c = x.shape #100, 1, 28,28 일 것. 
        x=x.reshape(-1,28*28) #100, 28*28 로 만드는 것.
        #x=x.reshape(b,-1) 따라서 이렇게 해도 같은 결과가 된다. 
        
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)

        return x
        
#모델, 로스, 옵티마이저
model = myMLP(hidden_size, num_classes).to(device)
loss = nn.CrossEntropyLoss() #분류 문제라서 이거 쓰는 게 답. 
optim = Adam(model.parameters(), lr=lr) 

#학습 루프.
for epoch in range(epochs):
    #mnist 데이터의 형태가 image, target 튜플임.
    for idx, (image, target) in enumerate(train_loader):
        image= image.to(device)
        target = target.to(device)
        
        out = model(image)
        #out= 모델의 출력, target = 정답
        loss_value = loss(out, target)
        
        #혹시라도 남아있는 애들을 지우고 백워드해준다. 이 순서가 더 낫다?
        optim.zero_grad() 
        
        loss_value.backward()
        optim.step()
        
        if idx % 100 ==0:
            print(loss_value.item())
        #item = 텐서의 값 1.23333.... 등을 뽑아냄.
