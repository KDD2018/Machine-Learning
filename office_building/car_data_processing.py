#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymongo
import math
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


def connect_mongo(projection):
    '''
    查询MongoDB数据
    projection 字段名称
    :return: data
    '''
    client = pymongo.MongoClient(
        'mongodb://spider:spider123@192.168.0.5:27017/')
    db = client['second_hand_car_autohome']
    collection = db['全国']
    data = pd.DataFrame(list(collection.find({}, projection)))
    return data


def conn_mysql(sql, col):
    '''
    sql 查询的SQL语句
    col 字段名称
    :return: data
    '''
    conn = pymysql.Connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!', db='valuation_web', port=3306,
                           charset='utf8')
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    conn.close()
    data = pd.DataFrame(list(data), columns=col)

    return data


def boxplot(data):
    '''
    根据箱线图取出异常值
    :param data: 数值型Series
    :return: 返回异常点及其位置
    '''
    plt.boxplot(x=data,  # 指定绘制箱线图的数据
                whis=3,  # 指定1.5倍的四分位差
                # notch=True,
                widths=0.8,  # 指定箱线图的宽度为0.8
                patch_artist=True,  # 指定需要填充箱体颜色
                showmeans=True,  # 指定需要显示均值
                boxprops={'facecolor': 'steelblue'},  # 指定箱体的填充色为铁蓝色
                # 指定异常点的填充色、边框色和大小
                flierprops={'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 2},
                # 指定均值点的标记符号（菱形）、填充色和大小
                meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markersize': 4},
                medianprops={'linestyle': '--', 'color': 'orange'},
                # 指定中位数的标记符号（虚线）和颜色
                labels=[''])  # 去除箱线图的x轴刻度值
    plt.show()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    low_whisker = Q1 - 3 * (Q3 - Q1)
    up_whisker = Q3 + 3 * (Q3 - Q1)
    # outlier = data[(data < low_whisker) | (data > up_whisker)]

    return low_whisker, up_whisker


def line_plot(data):
    '''
    通过3sigma原则去除异常值
    :param data: 数值型Series
    :return: low_whisker 异常下界
             up_whisker 异常上界
    '''
    plt.plot(
             data,  # y轴数据
             linestyle='-',  # 设置折线类型
             linewidth=2,  # 设置线条宽度
             color='steelblue',  # 设置折线颜色
             marker='o',  # 往折线图中添加圆点
             markersize=4,  # 设置点的大小
             markeredgecolor='black',  # 设置点的边框色
             markerfacecolor='black')  # 设置点的填充色
    # 添加上下界的水平参考线（便于判断异常点，如下判断极端异常点，只需将2改为3）
    low_whisker = data.mean() - 3 * data.std()
    up_whisker = data.mean() + 3 * data.std()
    plt.axhline(y=low_whisker, linestyle='--', color='gray')
    plt.axhline(y=up_whisker, linestyle='--', color='gray')
    plt.show()

    return low_whisker, up_whisker


def preprocess(data):
    '''
    数据预处理
    :param data: 要处理的数据框
    :return: 与处理后的数据框
    '''

    data['risk_27_check'] = data['risk_27_check'].fillna('None')
    data = data[data['risk_27_check'].isin(['None', '0'])]  # 去除重大事故的车
    data = data[data['sell_times'] < 5]  # 去除过户次数大于4次的二手车
    data['meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
    data['car_price'] = data['car_price'] / 10000.0  # 换算成万元
    data = data.dropna()
    # print(data.isnull().any())
    data['car_age'] = data['register_time'].map(lambda x: (datetime.now() - x).days / 365)
    data = data[data.car_type != '0']  # 去除无法确认的的车型

    return data


def feature_encode(data):
    '''
    特征离散编码
    :param data: 数据框
    :return: 特征编码后的数据框
    '''
    data['displacement'] = pd.cut(data.displacement, bins=[-1, 0.01, 1, 1.6, 2, 3, 4, 8],
                                  labels=['0L', '0.01-1L', '1-1.6L', '1.6-2L', '2-3L', '3-4L', '4L以上'])  # 排量
    data['car_age'] = pd.cut(data.car_age, bins=[-1, 1, 3, 5, 8, 50],
                             labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])  # 车龄
    data['sell_times'] = pd.cut(data.sell_times, bins=[-1, 0.1, 1.1, 2.1, 3.1, 4.1],
                                labels=['0次', '1次', '2次', '3次', '4次'])  # 过户次数

    return data


def write2csv(data, batch_size):
    '''
    将数据框分批次写入多个csv
    :param batch_size: 每批次写入样本数量
    '''
    epoch = math.ceil(df.shape[0] / batch_size)
    print('**********************开始写入CSV文件*****************************')
    for i in range(epoch):
        data = df[i * 50000: (i + 1)*50000]
        data.to_csv('/home/kdd/python/car/car_%d.csv'%i, encoding='gbk', chunksize=10000)  # 写入csv
    print('**********************完成CSV文件写入*****************************')


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

    res = prediction - y_test
    plt.scatter(y_test, res)
    plt.xlabel('y_test_true')
    plt.ylabel('residual')
    plt.show()


# np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)

# Mongodb查询文档字段
project = {'_id': 0, 'title': 1, 'car_address': 1, 'displacement': 1, 'emission_standard': 1,
           'is_have_strong_risk': 1, 'level': 1, 'meter_mile': 1, 'registe_time': 1, 'sell_times': 1,
           'semiautomatic_gearbox': 1, 'year_check_end_time': 1, 'car_price': 1}

# SQL查询语句
sql = """
select
	brand_name,
	vehicle_system_name,
	car_model_name,
	register_time,
	meter_mile,
	semiautomatic_gearbox,
	displacement,	
	sell_times,
	risk_27_check,
	`type`,
	car_class,
	price
from
	second_car_sell
"""

col = ['brand_name', 'car_system', 'car_model','register_time', 'meter_mile', 'gearbox', 'displacement', 'sell_times',
       'risk_27_check', 'car_type', 'car_class', 'car_price']
col1 = ['brand_name', 'car_system', 'car_model', 'gearbox', 'sell_times','displacement','car_age', 'meter_mile',
        'car_type', 'car_price']
# 获取Mongodb数据
# data = connect_mongo(project)
# data.to_csv('/home/kdd/Desktop/car.csv', encoding='gbk')  # 写入csv

car_class_dict = {'suv': ['大型SUV', '中大型SUV', '小型SUV', '紧凑型SUV', '中型SUV'],
                  'saloon': ['小型车', '大型车', '微型车', '中大型车', '中型车', '紧凑型车'], 'supercar': '跑车',
                  'pickup': ['皮卡', '微卡', '高端皮卡'], 'mpv': 'MPV', 'minibus':['轻客', '微面']}


if __name__ == '__main__':
    # 获取MySQL数据
    data = conn_mysql(sql, col)
    data = data[data.car_class=='saloon']  #  分别对每类车进行建模
    # print(data.head())

    # 数据预处理
    data = preprocess(data)

    # 特征编码
    data = feature_encode(data)

    # 去除异常
    # low_whisker, up_whisker =  boxplot(data['meter_mile'])
    # l, u = boxplot(data.car_price)
    # data = data[~(data.meter_mile > up_whisker) | (data.meter_mile < low_whisker)]
    # low_whisker, up_whisker =  line_plot(data['car_price'])
    # data = data[~(data.car_price > u) | (data.car_price < l)]

    # 有效字段
    df = data[col1]
    df.reset_index(drop=True)
    print(set(data.car_class))
    print(df.shape)

    # 将数据框分批次写入多个csv
    # write2csv(data=df, batch_size=50000)
    # print(set(df['car_type']))
    # df.to_csv('/home/kdd/python/car/car.csv', encoding='gbk', chunksize=10000)

    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(df)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 建立机器学习模型
    model_select('RF', X_train, X_test, y_train, y_test)  # 模型名称 ['SVR', 'CART', 'RF', 'GBR']






