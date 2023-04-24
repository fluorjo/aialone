from utils.getModules import getTransform
from utils.getModules import getTargetModel
from utils.parser import infer_parser_args
from utils.parser import load_trained_args
import torch
import os

from PIL import Image
import torch.nn.functional as F
def main():
    pass
    args=infer_parser_args()
    
    assert os.path.exists(args.folder),"학습 폴더 없음."
    assert os.path.exists(args.image),"추론할 이미지 없음."
        
    trained_args=load_trained_args(args)
    model = getTargetModel(trained_args)
    model.load_state_dict(torch.load(os.path.join(args.folder,'best_model.ckpt')))
    transform = getTransform(trained_args)
    
    img=Image.open(args.image)
    img=transform(img)
    img=img.unsqueeze(0)
    
    output=model(img)
    prob=F.softmax(output, dim=1)
    index=torch.argmax(prob)
    value=torch.max(prob)
    
    classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    print(f'Image is {classes[index]}, and the confidence is {value*100:.2f} %')
    
    
    print(output)
if __name__ == '__main__':
    main()
    
    