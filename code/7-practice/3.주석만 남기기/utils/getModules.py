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
        #resize->crop 해주기. img_size로. 
        transform = 
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
    #cifar 데이터 쓸 경우
    #dog dataset 쓸 경우 - image 폴더 혹은 커스텀셋
        #이미지 폴더
            #데이터셋의 위치 지정. 
        #커스텀1

        #커스텀2- train/test 나뉘어있지 않은 경우. 
            #tmp_dataset은 원본 데이터 그대로 가져온 것. 
            #train - test 나눠주기. train size는 0.8.
            
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
        #파인튜닝 할 경우
            #resnet18 모델과 웨이트 가져옴. 인수로 넣어줘야 함. 
            #모델의 최종 출력단을 변경함. 원래 1000개의 클래스 분류하게 돼있는 걸 5개 분류하도록 바꿈.
            
        #안 할 경우
        raise ValueError('no model implemented~')
    
    return model 