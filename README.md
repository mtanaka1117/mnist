# mnist
MNSIT model using pytorch.

## Environment
Tested on Ubuntu:22.04 & NVIDIA TITAN RTX  

| Framework             | version    |
| --------------------- | ---------- |
| CUDA                  | 12.0       |
| Python                | 3.10.13    |
| torch                 | 2.1.0      |
| torchvision           | 0.16.0     |


## Directory
```bash
.
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── finetune_dataset
├── test_dataset
│
├── model_50.pt
├── finetuned_model.pt
│
├── Dockerfile
├── run.sh
├── fine_tuning.py
├── model.py
├── predict.py
├── test.py
└── train.py
```

## Dataset
finetune_dataset:  
test_dataset: 100 images (10 for each number)  


## Setup
Setup with Dockerfile. (Edit the Dockerfile according to the availability of GPU or the CUDA version.)
```bash
git clone https://github.com/mtanaka1117/mnist.git
cd mnist
chmod +x ./run.sh
./run.sh
```

## Model
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```


## Train
Train and evaluate the model with MNIST dataset.
Model weights are stored in model_*.pt, and training loss is plotted in loss.png
```
python train.py --epochs num_epoch --batch_size batch_size
```

## Fine-tuning
Fine-tune the model with finetune_dataset.
Model weights are stored in finetuned_model.pt
```
python fine_tuning.py --epochs num_epoch --batch_size batch_size
```

## Test
Run test on the fine-tuned model with the original test dataset.
The result is plotted in confusion_matrix.png
```
python test.py --batch_size batch_size
```

## Predict
```
python predict.py
```

