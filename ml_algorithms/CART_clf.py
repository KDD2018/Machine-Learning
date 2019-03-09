#!/usr/bin/python3
# -*- coding: utf-8 -*-


from math import pow


class Node:
    '''
    树的节点
    '''
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea  # 用于切分数据集的属性的列索引
        self.value = value  # 设置划分的值
        self.results = results  # 存储叶节点所属的类别
        self.right = right  # 右子树
        self.left = left  # 左子树


def label_uniq_cnt(data):
    '''
    Desc: 统计数据集中不同类标签label的个数
    Args: data(list) 原始数据集
    Returns: label_uniq_cnt(int) 样本中标签的类别数
    '''
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x) - 1]  # 取得每个样本的类标签label
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] = label_uniq_cnt[label] + 1
    return label_uniq_cnt

def cal_gini_index(data):
    '''
    Desc: 计算Gini指数
    Args: data(list): 数据集
    Returns: Gini指数
    '''
    total_sample = len(data)
    if len(data) == 0:
        return 0
    label_counts = label_uniq_cnt(data)  # 统计不同标签的个数
    gini = 0
    for label in label_counts:
        gini = gini + pow(label_counts[label], 2)
    gini = 1- float(gini) / pow(total_sample, 2)
    return gini

def split_tree(data, fea, value):
    '''
    Desc: 根据特征fea的值value将数据集data划分成左右子树
    Args: data(list) 数据集
          fea(int) 待分割特征的索引
          value(float) 待分割特征的具体值
    Rerurns: 分割后的左右子树
    '''
    set1 = []
    set2 = []
    for x in data:
        if x[fea] >= value:
            set1.append(x)
        else:
            set2.append(x)
    return (set1, set2)

def build_tree(data):
    '''
    Desc: 构建CART分类树
    Args: data(list) 训练样本
    Returns: node 树的根节点
    '''
    # 构建决策树，返回决策树的根节点  
    if len(data) == 0:
        return Node()

    # 1. 计算当前的Gini指数
    currentGini = cal_gini_index(data)
    bestGain = .0
    bestCriteria = None  # 存储最佳切分属性以及最佳切分点
    bestSets = None  # 存储切分后的两个数据集

    feature_num = len(data[1]) - 1  # 特征数

    # 2. 找到最好的划分
    for fea in range(0, feature_num):
        # 取得特征fea的所有可能取值
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1  # 存储特征fea的所有可能取值

        # 针对可能取值，将数据集划分，并计算Gini指数
        for value in feature_values.keys():  # 遍历该属性的所有切分点
            # 根据特征fea的值value将数据集划分成左右子树
            (set1, set2) = split_tree(data, fea, value)
            # 计算当前Gini指数
            nowGini = float(len(set1) * cal_gini_index(set1) + len(set2) * cal_gini_index(set2)) / len(data)
            # 计算Gini指数增量
            gain = currentGini - nowGini
            # 判断此划分是否比当前的划分更好
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set1, set2)

    # 3. 判断划分是否结束
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return Node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return Node(results=label_uniq_cnt(data))

def predict(sample, tree):
    '''
    Desc: 预测
    Args: sample(list) 需要预测的样本
          tree(类) 构建好的分类树
    Returns: tree.results 所属类别
    '''

    # 1.只有树根
    if tree.results = None:
        return tree.results
    else:
    # 2.有左右子树
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)

