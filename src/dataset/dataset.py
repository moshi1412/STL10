import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
# print(label_one_hot)
classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
class STLDataset():
    def __init__(self, image_path,transform=None):
        self.image_path=image_path
        self.transform = transform 
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, index):
        file_info = self.image_path[index]
        file_name=file_info['image']
        label=file_info['label']
        label=classes.index(label)
        # print(label)
        image = Image.open(file_name)
        # print(image.size)#96*96
        
        if self.transform:
            image = self.transform(image)
        else :
            self.transform=transforms.Compose( [ transforms.ToTensor()])
            image=self.transform(image)
        return image,label



def get_file_name(class_name_path):
        #得到class_name_path下面的图片的路径
        file_names = os.listdir(class_name_path)
        return  [os.path.join(class_name_path,file_name) for file_name in file_names]
def get_datasets(root_dir, transform=None,split=0.8):
    labels=os.listdir(root_dir)
    data=[]
    for label in labels:
        class_name_path=os.path.join(root_dir,label)
        file_names=get_file_name(class_name_path)
        for file_name in file_names:
            data.append({'image':file_name,'label':label})
    # print(data)
    random.shuffle(data)
    data_len=len(data)
    # print(data_len)
    image_path_train=data[:int(data_len*split)]
    image_path_val=data[int(data_len*split):]
    train_set=STLDataset(image_path_train, transform=transform)
    val_set=STLDataset(image_path_val, transform=None)
    
    
    return train_set, val_set