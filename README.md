

# SSD.Pytorch

Pytorch implementation of [[SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325)]. 

this repository is heavily depend on this implementation [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).since orginal code is too old to fit the recent version of pytorch. I make some changes , fix some bugs, and give out SSD512  code.

## Environment

python3.7 (python3 may work ok)

pytorch1.3

opencv

## Dataset

Currently I only trained on Pascal VOC dataset and my own plate dataset.

U can make ur own dataset as VOC format and train ur own ssd model.

datasets are put under ./data , u should change the path according in voc0712.py.

## Train

First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

By default, we assume you have downloaded the file in the ./weights dir:

```shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

to train VOC or ur own dataset, simply run :

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --input 512 --dataset_root ./data/VOCdevkit --num_class 21 --num_epoch 300 --lr 0.001 --batch_size 16
```

or u can resume ur training from the checkpoint under dir ./weights/

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --input 512 --dataset_root ./data/VOCdevkit --num_class 21 --num_epoch 300 --lr 0.001 --batch_size 16 --resume ./weights/ssd512_VOC_12000.pth
```

## Evaluation

use the eval.py to eval ur model:

```
python eval.py --input 512 --trained_model weights/ssd512_VOC_28000.pth
```

## Demo

u can test single image using demo.py, just change a bit code in demo.py

<img src="/home/kong/Documents/ssd.pytorch/resut.jpg" alt="[](./resut.jpg)" style="zoom: 67%;" />

## Results

  VOC2007 test (0.5) results:

| model  | paper | this implements |
| ------ | ----- | --------------- |
| SSD300 | 77.2  | TD              |
| SSD512 | 79.8  | TD              |
|        |       |                 |

