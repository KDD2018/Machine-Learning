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


def connect_mongo(projection):
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
                flierprops={
                    'markerfacecolor': 'red',
                    'markeredgecolor': 'red',
                    'markersize': 2},
                # 指定均值点的标记符号（菱形）、填充色和大小
                meanprops={
                    'marker': 'D',
                    'markerfacecolor': 'black',
                    'markersize': 4},
                medianprops={
                    'linestyle': '--',
                    'color': 'orange'},
                # 指定中位数的标记符号（虚线）和颜色
                labels=['']  # 去除箱线图的x轴刻度值
                )
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
    plt.plot(data,  # x轴数据
             # data,  # y轴数据
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
	register_time,
	meter_mile,
	semiautomatic_gearbox,
	displacement,
	emission_standard,
	sell_times,
	year_check_end_time,
	is_have_strong_risk,
	risk_27_check,
	`type`,
	price
from
	second_car_sell
"""

col = ['brand_name', 'car_sys', 'register_time', 'meter_mile', 'gearbox', 'displacement', 'emission', 'sell_times',
       'year_check', 'strong_risk', 'risk_27_check', 'type', 'car_price']

# 获取Mongodb数据
# data = connect_mongo(project)
# data.to_csv('/home/kdd/Desktop/car.csv', encoding='gbk')  # 写入csv

# 获取MySQL数据
data = conn_mysql(sql, col)


# 数据预处理
data['risk_27_check'] = data['risk_27_check'].fillna('None')
data = data[data['risk_27_check'].isin(['None', '0'])]  # 去除重大事故的车
data = data[data['sell_times'] < 5]  # 去除过户次数大于4次的二手车
data['meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
data['car_price'] = data['car_price'] / 10000.0  # 换算成万元
data = data[data.car_price != data['car_price'].max()]  # 去除价格最大的跑车
data = data.dropna()
# print(data.isnull().any())
data['car_age'] = data['register_time'].map(lambda x: (datetime.now() - x).days / 365)
# data.loc[data['risk_27_check'] == '0', 'risk_27_check'] = '无'


# 去除异常
print(data.shape)
low_whisker, up_whisker =  line_plot(data['meter_mile'])
data = data[~(data.meter_mile > up_whisker) | (data.meter_mile < low_whisker)]
print(data['meter_mile'].max())
# outlier = boxplot(data['meter_mile'])
# print(outlier)
print(data.shape)
print(data.meter_mile[8000:10000])

# 特征编码
data['displacement'] = pd.cut(data.displacement, bins=[-1, 0.01, 1, 1.6, 2, 3, 4, 8],
                              labels=['0L', '0.01-1L', '1-1.6L', '1.6-2L', '2-3L', '3-4L', '4L以上'])  # 排量
data['car_age'] = pd.cut(data.car_age, bins=[-1, 1.01, 3.01, 5.01, 8.01, 50],
                         labels=['1年以内', '3年以内', '5年以内', '8年以内', '8年以上'])  # 车龄
data['sell_times'] = pd.cut(data.sell_times, bins=[-1, 1, 2, 3, 4, 10],
                            labels=['0次', '1次', '2次', '3次', '4次'])  # 过户次数


# 标准化
scale_data = StandardScaler().fit_transform(data[['meter_mile', 'car_price']])
data['meter_mile'] = scale_data[:, 0]
data['car_price'] = scale_data[:, 1]


# labelencoder = LabelEncoder() # 标签实例化
# data['sell_times'] = labelencoder.fit_transform(data['sell_times'])
# data['displacement_label'] = labelencoder.fit_transform(data['displacement'])
# data['gearbox_label'] = labelencoder.fit_transform(data['semiautomatic_gearbox'])
# data['carAge_label'] = labelencoder.fit_transform(data['car_age'])

col1 = ['register_time', 'emission', 'year_check', 'strong_risk', 'type', 'risk_27_check']
df = data.drop(columns=col1)  # 删除无效字段



# print(df.groupby('sell_times').count())

# 准备训练集和测试集的特征值、目标值
target = df['car_price']
feature_name = df[['brand_name', 'car_sys', 'displacement', 'meter_mile', 'sell_times', 'gearbox', 'car_age']]
feature = pd.get_dummies(feature_name)  # 哑变量编码
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)  # 直接划分训练集

# print(y_test.shape)


# 建立机器学习模型

# 支持向量机回归
# regressor = svm.SVR()

# CART回归树
# regressor = tree.DecisionTreeRegressor(max_depth=8)

# 随机森林
# regressor = RandomForestRegressor(max_depth=7, n_estimators=100)


# 梯度提升
# regressor = GradientBoostingRegressor()


# regressor = regressor.fit(X_train, y_train)
# score = regressor.score(X_test, y_test)
# prediction = regressor.predict(X_test)
# # print(regressor.feature_importances_)
# print(score)


# 残差图
# res = prediction - y_test
# print(res)
# plt.scatter(y_test, res)
# plt.xlabel('y_test_true')
# plt.ylabel('residual')
# plt.show()
