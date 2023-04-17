#3.데이터 로더. getDataLoader

#평균, 표준편차
from .tools import CIFAR_MEAN,CIFAR_STD
#transform
    #크기 변경
from torchvision.transforms import Resize
    #텐서로
from torchvision.transforms import ToTensor
    #정규화
from torchvision.transforms import Normalize 
    #위의 것들을 묶어서 할 수 있게 해주는 기능
from torchvision.transforms import Compose

#데이터셋
from torchvision.datasets import CIFAR10
#데이터 로더
from torch.utils.data import DataLoader

def getDataLoader(args):
    #평균, 표준편차
    mean=CIFAR_MEAN
    std=CIFAR_STD
    
    #transform
    transform=Compose([
        Resize((args.img_size,args.img_size)),
        ToTensor(),
        Normalize(mean,std)
    ])
    #데이터셋 삽입
    train_dataset=CIFAR10(root='./cifar',train=True, transform=transform, download=True)
    test_dataset=CIFAR10(root='./cifar',train=False, transform=transform, download=True)
    #데이터로더 삽입
    train_loader=DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False)
    
    return train_loader, test_loader

#6 모델 불러오기.
def getTargetModel(args):
    if args.model_type == 'mlp': 
        from networks.MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes).to(args.device)
    elif args.model_type == 'lenet': 
        from networks.LeNet import myLeNet
        model = myLeNet(args.num_classes).to(args.device) 
    elif args.model_type == 'linear':
        from networks.LeNet import myLeNet_linear
        model = myLeNet_linear(args.num_classes).to(args.device)  
    elif args.model_type == 'conv':
        from networks.LeNet import myLeNet_convs 
        model = myLeNet_convs(args.num_classes).to(args.device)  
    elif args.model_type == 'incep':
        from ..networks.LeNet import myLeNet_incep 
        model = myLeNet_incep(args.num_classes).to(args.device)  
    else : 
        raise ValueError('no model')
    return model
