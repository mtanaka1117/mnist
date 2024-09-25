import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Net
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
# model.load_state_dict(torch.load("model_50.pt"))

model.load_state_dict(torch.load("finetuned_model.pt"))
model.eval()


PATH = "./test_dataset/**/"
files = glob.glob(PATH + '*.jpg')

with torch.no_grad():
    for file in files:
        image = Image.open(file).convert('L')

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((28, 28)),
                                        transforms.Normalize((0.5,), (0.5,))])
        image = transform(image).to(device).unsqueeze(0)
        # image = transform(image).to(device)

        output = model(image)
        _, prediction = torch.max(output, 1)

        print("{} -> result = ".format(file) + str(prediction[0].item()))
