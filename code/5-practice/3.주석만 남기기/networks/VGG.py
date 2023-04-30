import torch.nn as nn

        #vgg는 conv 거칠 때 이미지 크기가 같아야 함. 
        #kernel size가 3이면 padding은 1, kernel size가 3 아니고 1이면 padding=0. 이러면 크기 유지됨.
   
        #처음과 마지막 아닌 conv들을 생성. 그래서 block의 전체 conv 수 -2 만큼 생성. 
        
        
        #마지막 conv면 kernel size가 1, 아닐 경우 3.
     

#VGG A,B,C,D,E가 유사한 형태를 가지고 있기 때문에 앞 부분을 상속하고 다른 부분만 바꿔줄 수 있음.    
