#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def load_data(file_name):
    '''
    Desc: 导入数据
    Args: file_name(string) 训练数据的位置
    Returns: feature_data(mat) 特征
                 label_data(mat) 标签
    '''
    f = open(file_name)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        label_tmp = []
        # strip去除字符串首尾字符(这里是空格)
        lines = line.strip().split('\t')
        feature_tmp.append(1)  # 偏置项
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))
        
        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data), np.mat(label_data)


def sig(x):
    '''
    Desc: Sigmoid 函数
    Args: x(mat): feature * w
    Returns: sigmoid(x)(mat):Sigmoid值
    '''
    return 1.0 / (1 + np.exp(-x))


def error_rate(h, label):
    '''
    Desc: 计算当前损失函数
    Args: h(mat) 预测值
          label(mat) 实际值
    Returns: err/m (float) 错误率
    '''
    m = np.shape(h)[0]
    sum_err = .0
    for i in range(m):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err += 0
    return sum_err/m


def lr_train_bgd(feature, label, maxCycel, alpha):
    '''
    Desc: 基于梯度下降法训练LR模型
    Args: feature(mat) 特征
          label(mat) 标签
          maxCycle(int) 最大迭代次数
          alpha(float) 学习速率（步长）
    Returns: w(mat) 权重
    '''
    n = np.shape(feature)[1]  # 特征个数
    w = np.mat(np.ones((n, 1)))  # 初始化权重
    i = 0
    while i <= maxCycel:  # 在最大迭代次数的范围内
        i += 1  # 当前迭代次数
        h = sig(feature * w)  # 计算sigmoid值
        err = label - h
        if i % 100 == 0:
            print('\t-----iter=' + str(i) + ', train error rate=' + str(error_rate(h, label)))
        w = w + alpha * feature.T * err  # 权重修正
    return w


def save_model(file_name, w):
    '''
    Desc: 保存模型
    Args: file_name(string) 模型保存的文件名
          w(mat) LR模型的权重
    '''
    m = np.shape(w)[0]
    f_w = open(file_name, 'w')
    w_array = []
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f_w.write('\t'.join(w_array))
    f_w.close()


if __name__ == "__main__":
    # 1.导入训练数据
    print('-------- 1. load data ---------')
    feature, label = load_data('data.txt')
    # 2.训练LR模型
    print('-------- 2. traing ---------')
    w = lr_train_bgd(feature, label, 1000, 0.01)
    # 3.保存最终模型
    print('-------- 3. save model ---------')
    save_model('weights', w)