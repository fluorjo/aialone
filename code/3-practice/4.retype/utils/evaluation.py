#7.평가
import torch

#정확도 계산
def eval(model, loader, args):
    
    #0에서 시작. 전체와 맞춘 것.
    total = 0 
    correct = 0 
    
    #데이터 디바이스로.
    for idx, (image, target) in enumerate(loader):
        image = image.to(args.device)
        target = target.to(args.device)

        #예측값 저장
        out = model(image)

        #가장 높은 값을 가진 걸 다른 변수에 저장.
        _, pred = torch.max(out, 1)

        #전체와 맞춘 것 누적해서 더함.
        correct += (pred == target).sum().item()
        total += image.shape[0]

    #맞춘 것 비율 리턴
    return correct / total
 
#클래스별 정확도 계산
def eval_class(model, loader, args):
    total = torch.zeros(args.num_classes) 
    correct = torch.zeros(args.num_classes) 
    for idx, (image, target) in enumerate(loader):
        image = image.to(args.device)
        target = target.to(args.device)

        out = model(image)
        _, pred = torch.max(out, 1)
        
        #클래스별 전체와 맞춘 것 누적.
        for i in range(args.num_classes): 
            correct[i] += ((target == i) & (pred == i)).sum().item()
            total[i] += (target == i).sum().item()
    return correct, total 