# 패키지 불러오기 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from utils.getModules import getDataLoader, getTargetModel
from utils.parser import parser_args
from utils.evaluation import eval, eval_class
from utils.tools import get_save_folder_path
import json
import sys
import os
sys.path.append(os.getcwd())

def main():
    args = parser_args() #파인 튜닝할 거라는 걸 명시해야 함. 
    
    #저장 폴더 설정. 
    save_folder_path = get_save_folder_path(args)
    
    os.makedirs(save_folder_path)
    with open(os.path.join(save_folder_path,'args.json'),'w') as f:
        json_args=args.__dict__.copy()
        del json_args['device']
        json.dump(json_args,f,indent=4)
        
    train_loader, test_loader=getDataLoader(args)

    # 모델, loss, optimizer 
    model = getTargetModel(args) # pytorch구현한 resnet18 + 뒤쪽에 MLP 변환 
    loss = nn.CrossEntropyLoss() 
    
    param_groups = [
        {'params':module, 'lr':args.lr * 0.01}
            for name, module in model.named_parameters() 
            if 'fc' not in name 
    ]
    param_groups.append({'params':model.fc.parameters(), 'lr':args.lr})
    
    optim = Adam(param_groups, lr=args.lr) # resnet의 기존 파라미터는 lr 0.00001, 뒤쪽에 붙은 MLP는 좀 큰 lr 
    
    # 학습 loop 
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

            if idx % args.save_itr == 0 : 
                print(loss_value.item())
                acc=eval(model, test_loader, args)
                print('accuracy :', acc) 
                #최고 정확도 갱신 시 모델 저장.
                if best_acc < acc:
                    best_acc=acc
                    torch.save(model.state_dict(), os.path.join(save_folder_path, f'best_model_{epoch}_{idx}.ckpt'))
                    
                    print(f'new best model acc:{acc*100:.2f}')
if __name__ == '__main__': 
    main()


