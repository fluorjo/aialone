from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from .tools import CIFAR_MEAN, CIFAR_STD

def getTransform(args): 
    if args.fine_tuning : 
        from torchvision.transforms._presets import ImageClassification
        transform = ImageClassification(crop_size=args.img_size, 
                                        resize_size=args.img_size)
    else : 
        mean = CIFAR_MEAN
        std = CIFAR_STD

        transform = Compose([
            Resize((args.img_size, args.img_size)), 
            ToTensor(),
            Normalize(mean, std)
        ])
    return transform

def getDataLoader(args): 
    transform=getTransform(args)
    
    if args.data=='cifar':
        train_dataset=CIFAR10(root='./cifar', train=True,transform=transform,download=True)
        train_dataset=CIFAR10(root='./cifar', train=False,transform=transform,download=True)

    else:
        if args.dataset=='imagefolder':
            from torchvision.datasets import ImageFolder
            train_dataset=ImageFolder(root='/home/dataset/dog_v1_TT/train',transform=transform)
            test_dataset=ImageFolder(root='/home/dataset/dog_v1_TT/test',transform=transform)

        elif args.dataset=='custom1':
            from utils.dogdataset import DogDataset
            train_dataset=DogDataset(root='/home/dataset/dog_v1_TT/train',transform=transform)
            test_dataset=DogDataset(root='/home/dataset/dog_v1_TT/test',transform=transform)
            pass
        
        elif args.dataset=='custom2':
            from utils.dogdataset import DogDataset
            from sklearn.model_selection import train_test_split
            
            tmp_dataset=DogDataset(root='/home/dataset/dog_v1',trans=transform)
            train_dataset,test_dataset=train_test_split(tmp_dataset,train_size=0.8,random_state=1111,shuffle=True)
            
    train_loader=DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True)
    
    return train_loader,test_loader
            
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
        from networks.LeNet import myLeNet_incep
        model = myLeNet_incep(args.num_classes).to(args.device) 
    elif args.model_type == 'vgg': 
        if args.vgg_type == 'a' : 
            from networks.VGG import VGG_A
            model = VGG_A(args.num_classes).to(args.device) 
        elif args.vgg_type == 'b' : 
            from networks.VGG import VGG_B
            model = VGG_B(args.num_classes).to(args.device) 
        elif args.vgg_type == 'c' : 
            from networks.VGG import VGG_C
            model = VGG_C(args.num_classes).to(args.device) 
        elif args.vgg_type == 'd' : 
            from networks.VGG import VGG_D
            model = VGG_D(args.num_classes).to(args.device) 
        elif args.vgg_type == 'e' : 
            from networks.VGG import VGG_E
            model = VGG_E(args.num_classes).to(args.device) 
    elif args.model_type == 'resnet': 
        #파인튜닝 한다면?
        if args.fine_tuning:
            import torch.nn as nn
            from torchvision.models import resnet18
            from torchvision.models import ResNet18_Weights
            #모델과 웨이트 가져옴. 
            weight=ResNet18_Weights
            model=resnet18(weight,progress=True)
            #모델의 최종 출력단을 변경함.

            # for idx, m in model.layer4.named_modules():
            #     if isinstance(m, nn.Conv2d):
            #         print(m,weight)
                
            model.fc=nn.Linear(512,5)
            model=model.to(args.device)
            pass
        #안 하면?
        else:
            from networks.ResNet import ResNet
            model = ResNet(args).to(args.device) 
    else : 
        raise ValueError('no model implemented~')
    
    return model 