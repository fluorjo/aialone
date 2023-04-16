import torch
import cv2
CIFAR_MEAN=[0.49139968, 0.48215827, 0.44653124]
CIFAR_STD=[0.24703233, 0.24348505, 0.26158768]

def draw_from_dataset(dataset,std,mean):
    #데이터셋 이미지 확인(텐서=>넘파이)
    def reverse_trans(x):
        x=(x*std)+mean
        #입력으로 들어오는 값들을 0-1 범위로 조정한 후 255 곱하기.
        return x.clamp(0,1)*255
    
    def get_numpy_image(data):
        #차원 순서를 재배치
        img=reverse_trans(data.permute(1,2,0)).type(torch.uint8).numpy()
        #BGR순서를 RGB로.
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
    #라벨=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #인덱스=1000
    idx=1000

    #데이터 가져오기. 
    img, label=dataset.__getitem__(idx)

    #이미지 크기 512로.
    img=cv2.resize(get_numpy_image(img),(512,512))

    #라벨 설정
    label=labels[label]
    cv2.imshow(label,img)