from torch.utils.data import Dataset
import os
from PIL import Image
class DogDataset(Dataset):
    def __init__(self,root,trans):
        super().__init__()
        self._classes=[c for c in os.listdir(root) if '.DS_Store' not in c]
        self.class2idx={c:i for i, c in enumerate(self._classes)}
        self.idx2class={i:c for i, c in enumerate(self._classes)}
        self.trans=trans
        self.image_pathes=[]
        
        for _class in self._classes:
            _class_path=os.path.join(root,_class)
            for image_name in os.listdir(_class_path):
                image_path=os.path.join(_class_path, image_name)
                self.image_pathes.append(image_path)
                
    def forward(self,x):
        return len(self.image_pathes)
    
    def __getitem__(self, idx):
        image_path=self.image_pathes[idx]
        image=Image.open(image_path).convert('RGB')
        image=self.trans(image)
        _class=os.path.basename(os.path.dirname(image_path))
        target=self.class2idx[_class]  
              
        return image,target