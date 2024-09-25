# mnist
This is a MNSIT model using pytorch.

## Environment
Tested on Ubuntu:22.04 & NVIDIA TITAN RTX  

| Framework             | version    |
| --------------------- | ---------- |
| CUDA                  | 12.0       |
| Python                | 3.10.13    |
| torch                 | 2.1.0      |
| torchvision           | 0.16.0     |


## Dataset

test_dataset: 100 images

## Setup
Setup with Dockerfile
```
git clone https://github.com/mtanaka1117/mnist.git
cd mnist
chmod +x ./run.sh
./run.sh
```

## Model
See model.py for the model.


## Train
```
python train.py
```

## Test 
```
python test.py
```

## Fine-tuning
```
python fine_tuning.py
```

