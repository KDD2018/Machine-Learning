#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymongo
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def read_csv(filelist):
    '''
    tensorflow读取csv
    :param filelist: 文件列表
    :return: 批次数据
    '''
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # file_queue = tf.data.Dataset.from_tensor_slices(filelist)

    # 2、 构造CSV阅读器，读取队列数据
    reader = tf.TextLineReader()
    # reader = tf.data.TextLineDataset()
    key, value = reader.read(file_queue)

    # 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
    records = [['None'], ['None'], ['None'], ['None'], ['None'], [0], [1.0], [0], [0]]
    sample = tf.decode_csv(records=value, record_defaults=records)

    # 4、批处理，读取多条数据
    sample_batch = tf.train.batch(sample, batch_size=100, num_threads=1, capacity=100)

    return sample_batch


def split_data(data):
    '''
    将数据划分为训练集和测试集
    :param data: 数据框
    :return: X_train, X_test, y_train, y_test
    '''
    target = data.iloc[:,-1]
    feature_X = data.iloc[:,:-1]
    feature = pd.get_dummies(feature_X)  # 哑变量编码
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)
    return X_train, X_test, y_train, y_test


def model_select(model_name, X_train, X_test, y_train, y_test):
    '''
    确定最优模型
    :param model_name: 模型名称 ['SVR', 'CART', 'RF', 'GBR']
    '''
    print('**************开始训练 %s 模型**************'%model_name)
    if model_name == 'SVR':
        regressor = svm.SVR()
    elif model_name == 'CART':
        regressor = tree.DecisionTreeRegressor(max_depth=8)
    elif model_name == 'RF':
        regressor = RandomForestRegressor(max_depth=8, n_estimators=100)
    else:
        regressor = GradientBoostingRegressor()
    regressor = regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print('\n**************拟合优度为：%f**************'%score)
    prediction = regressor.predict(X_test)
    # print(regressor.feature_importances_)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction, multioutput='uniform_average')
    print('\n**************均方误差为：%f**************' %mse)




if __name__ == '__main__':
    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(df)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 建立机器学习模型
    model_select('RF', X_train, X_test, y_train, y_test)  # 模型名称 ['SVR', 'CART', 'RF', 'GBR']