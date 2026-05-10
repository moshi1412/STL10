from tool.metric import Metric
import os
from dataset.dataset import *
import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
import numpy as np
log_dir = '../out/test/deeper'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'eval.log')),
        logging.StreamHandler()
    ]
)


ROOT_DIR='../STL10'
train_root=os.path.join(ROOT_DIR,'train')
test_root=os.path.join(ROOT_DIR,'test')
TRANSFORM=transforms.Compose([transforms.ToTensor()])
BATCH_SIZE=8
_ ,val_set = get_datasets(test_root, transform=TRANSFORM,split=0)
# _ = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
metric=Metric(10,label_name=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
# from model.SEBlock import CNN_SE,CNN_CBAM

# model=CNN_CBAM().to('cuda')
from model.deeper import CNNModel
model=CNNModel().to('cuda')
#加载模型权重
model.load_state_dict(torch.load(os.path.join('../out/train/deeper', 'best_model.pth')),strict=True)

model.eval()
for image,label in val_loader:
    image=image.to('cuda')
    label=label.to('cuda')
    output=model(image)

    pred = torch.argmax(output, dim=1)
    metric.__after_step__(pred,label)

metric.__after_epoch__()
metric.print_metric()
metric.reset()
