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
import seaborn as sn
import pandas as pd

from model import Net


def test(model, test_loader, device):
    correct = 0
    label_count = 0
    model.eval()
    
    cm = torch.zeros(10, 10)
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels.to(device)).sum().item()
            label_count += len(labels)
            
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
        print(f"test accuracy:{correct/label_count*100}%")
        
    
    # Draw confusion matrix
    cm = cm.numpy()
    for i in range(10):
        cm[i] = cm[i]/np.sum(cm[i])
    cm = np.around(cm,3)

    plt.figure(figsize=(10,7))
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sn.set(font_scale=1.3)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 12}, cmap='Blues')
    plt.suptitle('Confusion Matrix', fontsize=16)
    plt.savefig('confusion matrix.png')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("finetuned_model.pt"))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = ImageFolder('./test_dataset', transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

test(model, test_loader, device)