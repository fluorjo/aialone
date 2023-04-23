#학습 종료 후 모델 불러와서 새 데이터 입력 후 추론 결과를 도출하는 것. 

from utils.parser import infer_parser_args
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
    img=Image.open(args,image)
    img=transform(img)
    img=img.unsqueeze(0)
    
    #모델 출력'
    output=model(img)
    
    prob=F.softmax(output, dim=1)
    index=torch.argmax(prob)
    value=torch.max(prob)
    
    
    print(output)
#기본 구조 - 메인 함수    
if __name__ == '__main__':
    main()
    
    