#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 1/19/2022 7:55 PM
# @Author  : FengDiao
# @Email   : diaofeng698@163.com
# @File    : train_tensor_encoder.py
# @Describe:



import torch


def encoder(boxes, labels):
    '''
    根据label好的数据，组装真值tensor，用于训练
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 3x3x12
    '''
    # 划分的grid数量
    grid_num = 3
    # 初始化一个14*14*30的张量
    # 12代表2*5+2
    target = torch.zeros((grid_num, grid_num, 12))
    cell_size = 1. / grid_num
    # calculate wide and height of boundingbox
    wh = boxes[:, 2:] - boxes[:, :2]
    # calculate center of boundingbox
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        # 向上取整
        ij = (cxcy_sample / cell_size).ceil() - 1  #
        # x,y 对应过去是 列和行
        # tensor部分要反一下
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
        xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target


if __name__ == "__main__":
    # 两分类
    label = torch.zeros((2))
    label[0] = 1
    label[1] = 2
    box = torch.tensor([[378 / 640, 61 / 426, 599 / 640, 360 / 426],[260 / 640, 1 / 426, 596 / 640, 376 / 426]])
    encoder(box, label)



