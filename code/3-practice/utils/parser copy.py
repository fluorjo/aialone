#1.parser
import torch
import argparse

def parser_args():
    parser=argparse.ArgumentParser()
    #배치사이즈
    parser.add_argument("--batch_size",type=int,default=100)
    #히든 사이즈
    parser.add_argument("--hidden_size",type=int,default=50)
    #클래스 수
    parser.add_argument("--num_classes",type=int,default=10)                
    #러닝레이트
    parser.add_argument("--lr",type=float,default=0.001)
    #에폭
    parser.add_argument("--epochs",type=int,default=3)
    #기기
    parser.add_argument("--device",default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #이미지 크기
    parser.add_argument("--img_size",type=int,default=32)
    #모델 선택
    parser.add_argument("--model_type",type=str,default='lenet', choices=['mlp','lenet','linear','conv','incep'])
    
    return parser.parse_args()