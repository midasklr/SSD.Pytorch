#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: demo.py
@Author:kong
@Time: 2020年01月21日09时40分
@Description:
'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import VOC_CLASSES as labels
from ssd import build_ssd

image_path = './test/example.jpg'
weight_path = './weights/ssd300_VOC_100000.pth'
model_input = 300

net = build_ssd('test', model_input, 21)    # initialize SSD
net.load_weights(weight_path)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = cv2.resize(image, (model_input, model_input)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

top_k=10
detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)    #4个尺度的缩放系数
for i in range(detections.size(1)):          #遍历num_class
    j = 0
    while detections[0,i,j,0] >= 0.2:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        j+=1
        image = cv2.rectangle(image,(pt[0],pt[1]),(pt[2],pt[3]),(255,0,0),2)
        image = cv2.putText(image,display_txt,(pt[2],pt[1]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
cv2.imwrite('./test/resut.jpg',image)
