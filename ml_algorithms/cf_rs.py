#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np


def cos_sim(x, y):
    """
    余弦相似度
    :param x(mat): 以行向量的形式存储,可以是用户或者商品
    :param y(mat): 以行向量的形式存储,可以是用户或者商品
    :return: x和y之间的余弦相似度
    """
    dot_product = x * y.T
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T)
    return (dot_product / denominator)[0, 0]


def similarity(data):
    """
    计算矩阵中任意两行之间的相似度
    :param data: (mat) 用户-商品矩阵
    :return: 相似度矩阵
    """
    # 用户数量
    m = np.shape(data)[0]
    # 初始化相似度矩阵
    w = np.zeros((m, m))

    for i in range(m):
        for j in range(i, m):
            if i == j:
                w[i, j] = 0
            else:
                w[i, j] = cos_sim(data[i, :], data[j, :])
                w[j, i] = w[i, j]

    return w


def user_based_recommend(data, w, user):
    """
    基于用户相似性为用户user推荐商品
    :param data: (mat)用户商品矩阵
    :param w: (mat)用户之间的相似度
    :param user: (int)用户ID
    :return: 推荐列表
    """
    m, n = data.shape
    items_user = data[user, :]

    # 找到用户user没有互动的商品
    itemID_not = [i for i in range(len(items_user)) if items_user[i]==0]

    # 对没有互动过的商品进行预测
    predict = {}
    # 遍历当前user没有互动过的商品
    for x in itemID_not:
        # 找到所有用户对商品x的互动信息
        item = np.copy(data[:, x])
        # 遍历所有用户对当前商品的互动信息
        for i in range(m):
            if item[i] != 0:
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]

    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


def item_based_recommend(data, w, user):
    """
    基于商品相似度为用户user推荐商品
    :param data: 商品-用户矩阵
    :param w: (mat)商品与商品之间的相似性
    :param user: 用户ID
    :return: predict(list)推荐列表
    """
    m, n = data.shape
    item_user = data[:, user]
    itemID_not = [i for i in range(len(item_user)) if item_user[i] == 0]

    predict = {}
    for x in itemID_not:
        item = np.copy(item_user)
        for j in