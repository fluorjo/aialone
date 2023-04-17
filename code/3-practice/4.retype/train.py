#0.토치
import torch
import torch.nn as nn

#1.parser
from utils.parser import parser_args
#3.데이터 로더, 6.모델 불러오기
from utils.getModules import getDataLoader, getTargetModel
#7.평가
from utils.evaluation import eval, eval_class

#옵티마이저 - 아담
from torch.optim import Adam

#시스템에 폴더, 파일 경로 삽입.
import sys
import os
sys.path.append(os.getcwd())

def main():
    #1.parser
    args=parser_args()
    #3.데이터 로더
    train_loader, test_loader = getDataLoader(args)
    #6.모델
    model=getTargetModel(args)
    #로스
    loss=nn.CrossEntropyLoss()
    #옵티마이저
    optim=Adam(model.parameters(),lr=args.lr)
    
    #학습
    for epoch in range(args.epochs):
        #enumerate-인덱스, 데이터 가져오기
        for idx, (image, target) in enumerate(train_loader):
            image=image.to(args.device)
            target=target.to(args.device)
            #이미지를 모델에 넣어 출력 변수 만들기.
            out=model(image)
            #로스
            loss_value = loss(out, target)
            #기울기 초기화. 옵티마이저.
            optim.zero_grad()
            #역전파
            loss_value.backward()
            #파라미터 업데이트
            optim.step()
            
            if idx%100 ==0:
                print(loss_value.item())
                print('정확도:', eval(model,test_loader,args))
#main함수는 이것이 직접 실행될 때만 실행되도록 하기.      
if __name__=='__main__':
    main()