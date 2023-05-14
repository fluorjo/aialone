from torch.utils.data import Dataset
import os
from PIL import Image
    
#Dogdataset 클래스는 토치의 데이터셋을 상속받음.
#init, len, getitem 필요.
class DogDataset(Dataset):
    #root, transformation 을 파라미터로 가짐. 
    def __init__(self,root,trans):
        super().__init__()
        #class 정보 - class에 뭐가 있는지. 루트의 폴더 이름으로 불러옴. DS_Store를 빼야 함.
        self._classes=[c for c in os.listdir(root) if '.DS_Store' not in c]
        #class-index 정보 넣어주기,
        self.class2idx={c:i for i, c in enumerate(self._classes)}
        self.idx2class={i:c for i, c in enumerate(self._classes)}
        #객체 변수로 transformation 관련 함수를 선언.(get modules에서 정의한 걸 불러옴.)
        self.trans=trans
        #전체 데이터를 하나의 리스트로 넣어주기.
        self.image_pathes=[]
        #dict 상태의 인덱스, 클래스를 리스트에 넣어줌.
        #2중 반복문. 클래스 이름과 각 클래스의 이미지 파일명 가져오기.
        for _class in self._classes:
            _class_path=os.path.join(root,_class)
            for image_name in os.listdir(_class_path):
        #이미지 경로 - os.path.join 사용해서 문자열 결합해 하나의 경로로 만듦.
                image_path=os.path.join(_class_path, image_name)
                self.image_pathes.append(image_path)
                
    def forward(self,x):
        #image_pathes의 길이 리턴
        return len(self.image_pathes)
    
    def __getitem__(self, idx):
        #리스트 image_pathes에서 인덱스를 인덱싱해서 가져옴=이미지의 경로 가져옴.
        image_path=self.image_pathes[idx]
        #이미지 경로를 바탕으로 PIL 이미지 객체 생성.
        image=Image.open(image_path).convert('RGB')
        #사이즈와 같은 transformation 진행
        image=self.trans(image)
        #정답을 추출. class라고 해줄 건데 앞에 _ 붙여야 함. 
        #dirname을 사용해 경로를 불러오고, 그 경로에서 basename을 사용해 파일명'만' 가져와 클래스명 가져오기.
        _class=os.path.basename(os.path.dirname(image_path))
        #정답을 인덱스로 변환
        target=self.class2idx[_class]        
        
        return image,target