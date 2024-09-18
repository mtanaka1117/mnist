import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder

import timm


def train(model, train_loader, criterion, device):
    train_loss = 0.0
    num_train = 0
    
    for param in model.parameters():
        param.requires_grad = False  # すべての層を凍結

    model.fc = nn.Linear(512, 10)
    model.fc.to(device)
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # model 学習モードに設定
    model.train()
    # model.eval()

    for images, labels in tqdm(train_loader):
        # batch数を累積
        num_train += len(labels)
        
        # viewで1次元配列に変更
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # 勾配をリセット
        outputs = model(images) # 推論
        loss = criterion(outputs, labels) # lossを計算
        loss.backward() # 誤差逆伝播
        optimizer.step()  # パラメータ更新
        train_loss += loss.item() # lossを累積
        
    train_loss = train_loss/num_train
    return train_loss


def valid(model, valid_loader, criterion, device):
    valid_loss = 0.0
    num_valid = 0

    model.eval()

    # 評価の際に勾配を計算しないようにする
    with torch.no_grad():
        for images, labels in valid_loader:
            num_valid += len(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
        valid_loss = valid_loss/num_valid

    return valid_loss



def fine_tuning(model, train_loader, valid_loader, device, num_epochs):
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, device=device)
        valid_loss = valid(model, valid_loader, criterion, device=device)
        print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.5f}, val_Loss : {valid_loss:.5f}')
        
        train_loss_list.append(train_loss)
        # valid_loss_list.append(valid_loss)
    return train_loss_list, valid_loss_list


dir_path = './cropped_img/'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.RandomRotation(20), 
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.1307,), (0.3081,))])


dataset = ImageFolder(dir_path, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Net().to(device)
# model.load_state_dict(torch.load("model_50.pt"))

model = timm.create_model('resnet18', num_classes=10, in_chans=1).to(device)
model.load_state_dict(torch.load("resnet18.pt"))

criterion = nn.CrossEntropyLoss()

num_epochs = 20
fine_tuning(model, train_loader, valid_loader, device, num_epochs)

torch.save(model.state_dict(), "fine_tuned_model.pt")