import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Net


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("model_50.pt"))
model.eval()


PATH = "./img/0.jpg"
image = Image.open(PATH).convert('L')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((28, 28)),
                                transforms.Normalize((0.5,), (0.5,))])
image = transform(image).view(-1, 28*28).to(device)

output = model(image)
_, prediction = torch.max(output, 1)

print("{} -> result = ".format(PATH) + str(prediction[0].item()))
