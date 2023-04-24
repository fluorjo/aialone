import torch
import cv2
import os
CIFAR_MEAN=[0.49139968, 0.48215827, 0.44653124]
CIFAR_STD=[0.24703233, 0.24348505, 0.26158768]
    # dataset 이미지 확인하기 (tensor -> numpy) ,
def draw_from_dataset(dataset, std, mean,):
    def reverse_trans(x):
        x = (x * std) + mean
        return x.clamp(0, 1) * 255 

    def get_numpy_image(data): 
        img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    idx = 1000 

    img, label = dataset.__getitem__(idx)
    img = cv2.resize(
        get_numpy_image(img), 
        (512, 512)
    )
    label = labels[label]
    cv2.imshow(label,img),
#저장 폴더 생성 or 새 학습 결과 생성 시 1 더해가며 폴더 생성 ,
def get_save_folder_path(args):
    if not os.pth.exists(args.save_folder):
        os.makedirs(args.save_folder)
        new_folder_name='1'
    else:
        current_max_value=max([int(f) for f in os.listdir(args.save_folder)])
        new_folder_name=str(current_max_value+1)
        
    path=os.path.join(args.save_folder, new_folder_name)
    
    return path