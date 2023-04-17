from utils.parser import parser_args
from utils.getModules import getDataLoader

import torch
import torch.nn as nn
from torch.optim import Adam
from utils.getModules import getDataLoader, getTargetModel
from utils.evaluation import eval, eval_class

import sys
import os
sys.path.append(os.getcwd())

def main():
    args=parser_args()
    train_loader, test_loader = getDataLoader(args)
    
    model=getTargetModel(args)
    loss=nn.CrossEntropyLoss()
    optim=Adam(model.parameters(),lr=args.lr)
    
    for epoch in range(args.epochs):
        for idx, (image, target) in enumerate(train_loader):
            image=image.to(args.device)
            target=target.to(args.device)
            
            out=model(image)
            loss_value = loss(out, target)
            optim.zero_grad()
            loss_value.backward()
            optim.step()
            
            if idx%100 ==0:
                print(loss_value.item())
                print('정확도:', eval(model,test_loader,args))
                
if __name__=='__main__':
    main()