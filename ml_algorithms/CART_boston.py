#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
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
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, shuffle=False)


# 建立CART回归树
dct = DecisionTreeRegressor()
dct.fit(x_train, y_train)
score = dct.score(x_test, y_test)
print('拟合优度为：%f'%score) 
