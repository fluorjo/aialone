import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam

from torch.utils.data import DataLoader

batch_size=100
hidden_size=500
num_classes=10
lr=0.001
epochs=3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset=MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)
test_dataset=MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)

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

model=myMLP(hidden_size,num_classes).to(device)
loss=nn.CrossEntropyLoss()
optim=Adam(model.parameters(),lr=lr)

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

        