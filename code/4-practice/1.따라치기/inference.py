#학습 종료 후 모델 불러와서 새 데이터 입력 후 추론 결과를 도출하는 것. 
from utils.getModules import getTransform
from utils.getModules import getTargetModel
from utils.parser import infer_parser_args
from utils.parser import load_trained_args
import torch
import os

from PIL import Image
import torch.nn.functional as F
#기본 구조 - 메인 함수
def main():
    pass
    #추론 전용 파서 불러오기
    args=infer_parser_args()
    
    assert os.path.exists(args.folder),"학습 폴더 없음."
    assert os.path.exists(args.image),"추론할 이미지 없음."
        
    #학습이 된 폴더 기반으로 학습된 args 불러오기
    trained_args=load_trained_args(args)
    #모델을 학습된 상황에 맞게 재설정
    model = getTargetModel(trained_args)
    #모델 가중치 업데이트
    model.load_state_dict(torch.load(os.path.join(args.folder,'best_model.ckpt')))
    #데이터 전처리 코드준비.
    transform = getTransform(trained_args)
    
    #이미지 불러오기
    img=Image.open(args.image)
    #모델에 넣기 위해 transform
    img=transform(img)
    #배치를 넣어줘야 함. 차원 낮춰줘야 함. 배치 1로 만듦.
    img=img.unsqueeze(0)
    
    #모델 출력'
    output=model(img)
    #스코어를 확률로.
    prob=F.softmax(output, dim=1)
    index=torch.argmax(prob)
    value=torch.max(prob)
    
    classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    print(f'Image is {classes[index]}, and the confidence is {value*100:.2f} %')
    
    
    print(output)
#기본 구조 - 메인 함수    
if __name__ == '__main__':
    main()
    
    