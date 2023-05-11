from torch.utils.data import Dataset
import os
from PIL import Image
class DogDataset(Dataset):
    def __init__(self,root,trans):
        super().__init__()
        #class가 뭐가 있는지. class 정보.
        self._classes=[c for c in os.listdir(root) if '.DS_Store' not in c]
        #class - index 정보 넣어주기.
        self.class2idx={c: i for i, c in enumerate(self._classes)}
        self.idx2class={i: c for i, c in enumerate(self._classes)}
        #객체 변수로 transformation 관련 함수를 선언해줌.(get modules애서 정의해온 걸 불러옴,)
        self.trans=trans
        #전체 데이터를 하나의 리스트(A)로 넣어주기.
        self.image_pathes=[]
        for _class in self._classes:
            _class_path=os.path.join(root,_class)
            for image_name in os.listdir(_class_path):
                image_path=os.path.join(_class_path,image_name)
                self.image_pathes.append(image_path)
        
        
    def __len__(self):
        #A의 길이를 리턴하기.
        return len(self.image_pathes)
    
    def __getitem__(self, idx):
        #리스트 A에서 인덱스를 인덱싱해서 가져오면
        image_path=self.image_pathes[idx]
        #이미지의 경로가 나오는데, 그 경로를 바탕으로 PIL 이미지 객체를 생성.
        image=Image.open(image_path).convert('RGB')
        #png=R,G,B에 더해서 알파라는 투명도 담당이 있음. 그걸 없애줘야 함.
        #사이즈와 같은 transformation을 진행한다. 
        image= self.trans(image)
        #이 이미지의 정답이 뭔지 뽑아야 함.
        #os.sep
        #_class= self.image_pathes[idx].split(os.sep)[-2]
        #위도 한 방법. 근데 운 안 좋게 파일명에 /나 \가 있을 수 있음.
        _class=os.path.basename(os.path.dirname(image_path))
        #정답을 인덱스로 변환.
        target=self.class2idx[_class]
        return image,target#이거 순서 바꾸면 다른 파일에서 함수에서의 순서도 바꿔줘야 함.