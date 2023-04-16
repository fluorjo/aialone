import torch
import cv2
CIFAR_MEAN=[0.49139968, 0.48215827, 0.44653124]
CIFAR_STD=[0.24703233, 0.24348505, 0.26158768]

def draw_from_dataset(dataset,std,mean):
    def reverse_trans(x):
        x=(x*std)+mean
        return x.clamp(0,1)*255