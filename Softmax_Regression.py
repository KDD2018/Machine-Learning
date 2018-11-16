#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def load_data(inputfile):
    '''
    Desc: 导入训练数据
    Args: inputfile(string) 训练样本的位置
    Returns: feature_data(mat) 特征
             label_data 标签
             k (int)  类别个数
    '''
    f = open(inputfile)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        feature_tmp.append(1)
        lines = line.strip().split('\t')
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))
        

def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    '''
    Desc: 梯度下降法训练Softmax模型
    Args: feature_data(mat) 特征
          label_data(mat) 标签
          k(int) 类别个数
          maxCycle(int) 最大迭代次数
          alpha(float) 学习率
    Returns: weights 权重
    '''
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        if i %100 == 0:
            print('\t------iter: ', i, \
                 ', cost: ', cost(err, label_data))
    rowsum = -err.sum(axis=1)
    rowsum = rowsum.repeat(k, axis=1)
    err = err / rowsum
    for x in range(m):
        err[x, label_data[x, 0]] += 1
    weights = weights + (alpha / m) * feature_data.T * err
    i += 1
    return weights


def cost(err, label_data):
    '''
    Desc: 计算损失函数
    Args: err(mat) exp的值
          label_data(mat) 标签值
    Returns: sum_cost / m(float) 损失函数的值
    '''
    m = np.shape(err)[0]
    sum_cost = .0
    for i in range(m):
        if err[i ,label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


def save_model(file_name, weights):
    '''
    Desc: 保存模型
    Args: file_name(string) 保存的文件名
          weights(mat) softmax 模型
    '''
    f_w = open(file_name, 'w')
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w+tmp.append(str(weights[i, j]))
        f_w.write('\t'.join(w_tmp) + '\n')
    f_w.close()


if __name__= '__main__':
    print('-------- 1. load data ---------')
    feature, label, k  = load_data(input_file)
    print('--------- 2. traing ------------')
    weights = gradientAscent(feature, label, k, 10000, 0.4)
    print('--------- 3. save model ------------')
    save_model('weights', weights)
    
