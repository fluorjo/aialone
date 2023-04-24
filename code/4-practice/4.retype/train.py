# 패키지 불러오기 
import torch
import torch.nn as nn
from utils.tools import get_save_folder_path
from utils.parser import parser_args
import json
from utils.getModules import getDataLoader
from utils.getModules import getTargetModel
from torch.optim import Adam
import os
import sys    
sys.path.append(os.getcwd())

def main():
    args=parser_args()
    #저장 폴더 설정. 
    save_folder_path=get_save_folder_path(args)
    
    os.makedirs(save_folder_path)
    #경로 지정 후 'w' 모드로 열어준다. 
    with open(os.path.join (save_folder_path, 'args.json'),'w') as f:
        #args를 dict로 바꾼다. 
        json_args=args.__dict__.copy()
        del json_args['device']
        #json.dump=파이썬 객체를 json 파일로 저장함. 
        json.dump(json_args,f,indent=4)
    train_loader,test_loader=getDataLoader(args)
    # 모델, loss, optimizer 
    model=getTargetModel(args)
    loss=nn.CrossEntropyLoss()
    optim=Adam(model.parameters(),lr=args.lr)
    # 학습 loop 
    #앞으로 정확도가 높아질수록 갱신이 될테니 우선 0으로 둔다. 
    best_acc=0
    #epoch - 0에서 시작 혹은 마지막 체크포인트 epoch에서 시작.

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        args.epochs = checkpoint['epoch']
        
    for epoch in range(args.epochs):
        for idx, (image, target) in enumerate(train_loader):
            image=image.to(args.device)
            target=target.to(args.device)
            out=model(image)
            loss_value=loss(out,target)
            optim.zero_grad()
            loss_value.backward()
            optim.step()
            if idx %100 ==0:
                print(loss_value.item())
                acc=eval(model, test_loader, args)
                print('accuracy=', acc)
                #최고 정확도 갱신 시 모델 저장.
                if best_acc < acc:
                    state={
                        'model':model.state_dict(),
                        'acc':acc,
                        'epoch':epoch,
                    },
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoin')
                    torch.save(state,'./checkpoint/ckpt.pth')
                    best_acc=acc
                    #학습시 각 레이어마다 텐서로 매핑되는 매개변수를 dict 형태로 저장.
                    #dict로 저장하는 이유? key-value형태가 돼야 해서? 나중에 불러올 때도 그렇게 불러오게 됨.
                    torch.save(model.state_dict(), os.path.join(save_folder_path, f'best_model_{epoch}_{idx}.ckpt'))
                    print(f'new best model acc:{acc*100:.2f}')
if __name__=='__main__':
    main()

