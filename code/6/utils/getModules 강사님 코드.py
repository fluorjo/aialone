from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from .tools import cifar_mean, cifar_std

def getTransform(args): 
    mean = cifar_mean
    std = cifar_std

    transform = Compose([
        Resize((args.img_size, args.img_size)), 
        ToTensor(),
        Normalize(mean, std)
    ])
    return transform

def getDataLoader(args): 
    transform = getTransform(args)

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
        if args.res_config == '18' : 
            from networks.ResNet import ResNet
            model = ResNet(args).to(args.device) 
        # elif args.res_config == 'b' : 
        #     from networks.VGG import VGG_B
        #     model = VGG_B(args.num_classes).to(args.device) 
        # elif args.res_config == 'c' : 
        #     from networks.VGG import VGG_C
        #     model = VGG_C(args.num_classes).to(args.device) 
        # elif args.res_config == 'd' : 
        #     from networks.VGG import VGG_D
        #     model = VGG_D(args.num_classes).to(args.device) 
        # elif args.res_config == 'e' : 
        #     from networks.VGG import VGG_E
        #     model = VGG_E(args.num_classes).to(args.device) 

    else : 
        raise ValueError('no model implemented~')
    
    return model 