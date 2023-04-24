import json 
import torch
import argparse
import os
def parser_args(): 
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--img_size", type=int, default=32)
    
    parser.add_argument("--model_type", type=str, default='lenet',choices=['mlp','lenet','linear','conv','incep'])
    parser.add_argument("--save_folder", type=str, default='code/4-practice/results')
    
    return parser.parse_args()
#parser에 argument 추가. 폴더, 이미지, 디바이스.
def infer_parser_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=str)
    parser.add_argument("--image",type=str)
    parser.add_argument("--device",default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #resume 추가. 
    #store_true= 어쨌든 arg가 있기만 하면 true 반환한다?
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    return parser.parse_args()

#학습된 arg 불러오기.
def load_trained_args(args):
    #args가 저장된 폴더에서 args.json 불러오기.
    with open(os.path.join(args.folder, 'args.json'),'r') as f:
        trained_args = json.load(f)

    #디바이스 설정.
    trained_args['device']=args.device
    #namespace=특정 객체를 이름에 따라 구분할 수 있는 범위(?)
    # ** =딕셔너리의 value 값을 넣는다?
    trained_args=argparse.Namespace(**trained_args)
    return trained_args