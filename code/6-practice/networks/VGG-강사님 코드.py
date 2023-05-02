import torch.nn as nn 

class VGG_conv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3): 
        super().__init__() 
        padding = 1 if kernel_size==3 else 0 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 

class VGG_Block(nn.Module):
    def __init__(self, in_channel, output_channel, num_convs, last_1conv=False): 
        super().__init__()
        self.first_conv =  VGG_conv(in_channel, output_channel)
        self.middle_convs = nn.ModuleList([
            VGG_conv(output_channel, output_channel) for _ in range(num_convs-2)
        ])
        kernel_size = 1 if last_1conv else 3 
        self.last_convs = VGG_conv(output_channel, output_channel, kernel_size=kernel_size)
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.middle_convs(x) 
        for module in self.middle_convs: 
            x = module(x) 
        x = self.mp(x)
        return x 
    
class VGG_classifier(nn.Module):
    def __init__(self, num_classes): 
        super().__init__() 
        self.fc = nn.Sequential(
            nn.Linear(25088, 4096), 
            nn.Linear(4096, 4096), 
            nn.Linear(4096, num_classes),  
        )

    def forward(self, x):
        x = self.fc(x)
        return x 

class VGG_A(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__() 
        self.VGG_Block1 = VGG_Block(3, 64, 1)
        self.VGG_Block2 = VGG_Block(64, 128, 1)
        self.VGG_Block3 = VGG_Block(128, 256, 2)
        self.VGG_Block4 = VGG_Block(256, 512, 2)
        self.VGG_Block5 = VGG_Block(512, 512, 2)
        self.FC = VGG_classifier(num_classes)
    
    def forward(self, x): 
        b, c, w, h = x.shape
        
        x = self.VGG_Block1(x)
        x = self.VGG_Block2(x)
        x = self.VGG_Block3(x)
        x = self.VGG_Block4(x)
        x = self.VGG_Block5(x)

        x = x.reshape(b, -1)
        x = self.FC(x)
        return x 


class VGG_B(VGG_A): 
    def __init__(self, num_classes): 
        super().__init__(num_classes) 
        self.VGG_Block1 = VGG_Block(3, 64, 2)
        self.VGG_Block2 = VGG_Block(64, 128, 2)


class VGG_C(VGG_B): 
    def __init__(self, num_classes): 
        super().__init__(num_classes) 
        self.VGG_Block3 = VGG_Block(128, 256, 3, last_1conv=True)
        self.VGG_Block4 = VGG_Block(256, 512, 3, last_1conv=True)
        self.VGG_Block5 = VGG_Block(512, 512, 3, last_1conv=True)


class VGG_D(VGG_B): 
    def __init__(self, num_classes): 
        super().__init__(num_classes) 
        self.VGG_Block3 = VGG_Block(128, 256, 3)
        self.VGG_Block4 = VGG_Block(256, 512, 3)
        self.VGG_Block5 = VGG_Block(512, 512, 3)

class VGG_E(VGG_D): 
    def __init__(self, num_classes): 
        super().__init__(num_classes) 
        self.VGG_Block3 = VGG_Block(128, 256, 4)
        self.VGG_Block4 = VGG_Block(256, 512, 4)
        self.VGG_Block5 = VGG_Block(512, 512, 4)
