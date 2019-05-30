#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor


# 获取Boston房价数据
boston = load_boston()
# 特征数据
feature = pd.DataFrame(boston.data, columns=boston.feature_names)
print(set(feature['CHAS']))
feature['CHAS'] = feature['CHAS'].astype('object')
feature = pd.get_dummies(feature)
print(feature.head())
# 目标值
target = boston.target


# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)


# 建立CART回归树
dct = DecisionTreeRegressor(max_depth=6)
for i in range(500):
    # dct.fit(x_train, y_train)
    # score = dct.score(x_test, y_test)
    kf = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(dct, feature, target, cv=kf)
    print('第%d次拟合优度为:%f'%(i, score.mean())) 
