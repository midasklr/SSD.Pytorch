# -*- coding: utf-8 -*-
# @Time : 2020/1/22 上午1:08 
# @Author : kong 
# @File : vgg16weights.py 
# @Software: PyCharm
import torch
vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')

print('VGG16 weights key:',vgg_weights.keys())
print('VGG16 weights include layers:',len(vgg_weights.keys()))

#vgg16_reducedfc
print('VGG16 layer1:',len(vgg_weights.values()))
print('VGG16 first layer1 Conv1_1(0.weight) shape:',vgg_weights['0.weight'].shape)
print('VGG16 second layer2 Conv1_1(0.weight) shape:',vgg_weights['0.bias'].shape)
for i, key in enumerate(vgg_weights.keys()):
    print('VGG16 layer {} : {} parameters shape {}'.format(i, key, vgg_weights[key].shape))