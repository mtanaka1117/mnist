import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Net

def test(model, test_loader, device):
    correct = 0
    label_count = 0
    model.eval()
    
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels.to(device)).sum().item()
        label_count += len(labels)
    print(f"test accuracy:{correct/label_count*100}%")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("finetuned_model.pt"))

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                                            shuffle=False)

test(model, test_loader, device)