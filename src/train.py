import os
from dataset.dataset import *
import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
import numpy as np
log_dir = '../out/train/transform'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'train.log')),
        logging.StreamHandler()
    ]
)

ROOT_DIR='../STL10'
train_root=os.path.join(ROOT_DIR,'train')
test_root=os.path.join(ROOT_DIR,'test')
TRANSFORM=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)
BATCH_SIZE=8
EPOCHS=200
# LR=0.001
# RESUME=True
train_set, val_set = get_datasets(train_root, transform=TRANSFORM,split=0.8)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# from model.SEBlock import CNN_SE,CNN_CBAM
# from model.deeper import CNNModel

# model=CNN_SE().to('cuda')
# model=CNN_CBAM().to('cuda')
# if RESUME:
#     model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')),strict=False)
from model.res import CNNResModel

model=CNNResModel().to('cuda')

# from model.deeper import CNNModel

# model=CNNModel().to('cuda')

criterion=torch.nn.CrossEntropyLoss()
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=5,
    eta_min=1e-5
)

epoch_losses = []
epoch_accuracies = []
best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    step = 0
    loss_list = []
    
    for image, label in train_loader:
        image = image.to('cuda')
        label = label.to('cuda')
        optimizer.zero_grad()
    
        output = model(image)
        loss = criterion(output, label)
        loss_list.append(loss.item())
    
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            avg_loss = sum(loss_list) / len(loss_list)
            logging.info(f'Epoch {epoch}, Step:{step}, Avg Loss: {avg_loss}')
            loss_list = [] 
            
        step += 1

    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
    epoch_losses.append(avg_loss)

    model.eval()
    accuracy=0
    for image,label in val_loader:
        image=image.to('cuda')
        label=label.to('cuda')
        output=model(image)
        pred = torch.argmax(output, dim=1)
        accuracy += (pred == label).sum().item()
    accuracy = accuracy / len(val_set)
    epoch_accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        logging.info(f'Epoch {epoch}, New best accuracy: {best_accuracy:.4f}, model saved')

    logging.info(f'Epoch {epoch}, Val Accuracy: {accuracy:.4f}')

np.save(os.path.join(log_dir, 'epoch_losses.npy'), np.array(epoch_losses))
np.save(os.path.join(log_dir, 'epoch_accuracies.npy'), np.array(epoch_accuracies))
