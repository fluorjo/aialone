from .tools import CIFAR_STD,CIFAR_MEAN
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.datasets import CIFAR10
import sys
import os
sys.path.append(os.getcwd())

def getTransform(args):
        return transform

def getDataLoader(args):
    mean=CIFAR_MEAN
    std=CIFAR_STD

    transform = Compose([
        Resize((args.img_size, args.img_size)), 
        ToTensor(),
        Normalize(mean, std)
    ])

    train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    # dataloader 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader

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
        raise ValueError('no model implemented~')
    return model