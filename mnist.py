import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download = True)
#検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download = True)
