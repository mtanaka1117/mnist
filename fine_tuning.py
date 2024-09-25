import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from model import Net


dir_path = './finetune_dataset/'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.RandomRotation(20), # Data Augumentation
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.1307,), (0.3081,))])

dataset = ImageFolder(dir_path, transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("model_50.pt")) # load weights

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

model.fc2 = nn.Linear(128, 10)
model.fc2.to(device)

optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model.train()

num_epochs = 50
for epoch in range(num_epochs):
    train_loss = 0.0
    num_train = 0
    
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
    print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.5f}')

torch.save(model.state_dict(), "finetuned_model.pt")