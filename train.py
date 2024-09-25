import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap

from model import Net

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()


def setup_all_seed(seed=0):
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0
    num_train = 0
    model.train()

    for images, labels in tqdm(train_loader):
        num_train += len(labels)
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss/num_train
    return train_loss


def valid_model(model, valid_loader, criterion, device):
    valid_loss = 0.0
    num_valid = 0
    model.eval()

    with torch.no_grad():
        for images, labels in valid_loader:
            num_valid += len(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
        valid_loss = valid_loss/num_valid

    return valid_loss


def run(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        valid_loss = valid_model(model, valid_loader, criterion, device=device)

        print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.5f}, val_Loss : {valid_loss:.5f}')
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    return train_loss_list, valid_loss_list


def plot_loss_graph(train_loss_list, test_loss_list):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    ax.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')

    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('loss', fontsize='20')
    ax.set_title('training and validation loss', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')

    plt.savefig('loss.png')


def test(model, test_loader, device):
    correct = 0
    label_count = 0
    model.eval()
    
    for images, labels in tqdm(test_loader):
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels.to(device)).sum().item()
        label_count += len(labels)
    print(f"test accuracy:{correct/label_count*100}%")


def feature_map(model, valid_loader, device):
    model.fc3 = nn.Identity()
    model.eval()
    
    features = None
    classes = None
    
    for images, labels in valid_loader:
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)

        if classes is None:
            classes = labels.cpu()
        else:
            classes = torch.cat((classes, labels.cpu()))

        if features is None:
            features = outputs.cpu()
        else:
            features = torch.cat((features, outputs.cpu()))
    
    _umap = umap.UMAP(random_state=42)
    X_umap = _umap.fit_transform(features)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=classes)
    plt.savefig('umap.png')


dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                        transform=transforms.ToTensor(),
                                        download = True)

# Divide into train and valid dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transforms.ToTensor(),
                                            download = True)

setup_all_seed()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loss_list, valid_loss_list = run(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device=device)

torch.save(model.state_dict(), "model_{}.pt".format(args.epochs)) # save model
plot_loss_graph(train_loss_list, valid_loss_list)

test(model, test_loader, device)
