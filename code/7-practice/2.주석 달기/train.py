# 패키지 불러오기 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from utils.getModules import getDataLoader
from utils.getModules import getTargetModel
from utils.parser import parser_args
from utils.evaluation import eval, eval_class
from utils.tools import get_save_folder_path
import json
import sys
import os
sys.path.append(os.getcwd())

def main():
    args = parser_args() 
    #저장 폴더 설정. 
    save_folder_path=get_save_folder_path(args)
    os.makedirs(save_folder_path)
    #경로 지정 후 'w' 모드로 열어준다. 
    with open(os.path.join(save_folder_path,'args.json'),'w') as f: 
        #args를 dict로 바꾼다. 
        json_args=args.__dict__.copy() 
        del json_args['device']
        #json.dump=파이썬 객체를 json 파일로 저장함. 
        json.dump(json_args,f,indent=4)
    train_loader, test_loader=getDataLoader(args)
    # 모델, loss, optimizer 
    # 모델은 기본 pytorch구현한 resnet18에다 뒤쪽에 MLP 변환 
    model=getTargetModel(args)
    loss = nn.CrossEntropyLoss() 
    
    # resnet의 기존 파라미터는 lr 0.00001, 뒤쪽에 붙은 MLP는 좀 큰 lr 
    param_groups = [
        {'params':module, 'lr':args.lr * 0.01}
            for name, module in model.named_parameters() 
            if 'fc' not in name 
    ]
    #모델 마지막 fc의 lr을 0으로 지정.
    param_groups.append({'params':model.fc.parameters(), 'lr':args.lr})    
    
    optim = Adam(model.parameters(), lr=args.lr)
    # 학습 loop 
    #앞으로 정확도가 높아질수록 갱신이 될테니 우선 0으로 둔다. 
    best_acc=0
    for epoch in range(args.epochs): 
        for idx, (image, target) in enumerate(train_loader): 
            image = image.to(args.device)
            target = target.to(args.device)           
            out = model(image)
            loss_value = loss(out, target)
            optim.zero_grad() 
            loss_value.backward()
            optim.step()
            
            #dataset 추가하면서 수정. save_itr라는 arg 추가해서 출력 횟수 조절. 
            if idx % args.save_itr == 0 : 
                print(loss_value.item())
                acc=eval(model, test_loader, args)
                print('accuracy :', acc) 
                #최고 정확도 갱신 시 모델 저장.
                if best_acc < acc:
                    best_acc=acc
                    #학습시 각 레이어마다 텐서로 매핑되는 매개변수를 dict 형태로 저장.
                    #dict로 저장하는 이유? key-value형태가 돼야 해서? 나중에 불러올 때도 그렇게 불러오게 됨.
                    torch.save(model.state_dict(), os.path.join(save_folder_path, f'best_model_{epoch}_{idx}.ckpt'))      
                    print(f'new best model acc:{acc*100:.2f}')
if __name__ == '__main__': 
    main()


