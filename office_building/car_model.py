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
data = data.dropna()

# print(data.isnull().any())
data['car_age'] = data['register_time'].map(
    lambda x: (datetime.now() - x).days / 365)
data['meter_mile'] = data['meter_mile'] / 10000.0
data['car_price'] = data['car_price'] / 10000.0
data.loc[data['risk_27_check'] == '0', 'risk_27_check'] = 0
data.loc[data['risk_27_check'] != 0, 'risk_27_check'] = 1
# print(data['car_price'])

# 特征编码
data['displacement'] = pd.cut(data.displacement, bins=[-1, 0.01, 1, 1.6, 2, 3, 4, 8],
                              labels=['0L', '0.01-1L', '1-1.6L', '1.6-2L', '2-3L', '3-4L', '4L以上'])  # 排量
data['car_age'] = pd.cut(data.car_age, bins=[-1, 1.01, 3.01, 5.01, 8.01, 50],
                         labels=['1年以内', '3年以内', '5年以内', '8年以内', '8年以上'])  # 车龄
data['sell_times'] = pd.cut(data.sell_times, bins=[-1, 1, 3, 5, 8, 20],
                            labels=['0次', '2次以内', '4次以内', '7次以内', '7次以上'])  # 过户次数
# print(data.head())
# labelencoder = LabelEncoder() # 标签实例化
# data['sell_times'] = labelencoder.fit_transform(data['sell_times'])
# data['displacement_label'] = labelencoder.fit_transform(data['displacement'])
# data['gearbox_label'] = labelencoder.fit_transform(data['semiautomatic_gearbox'])
# data['carAge_label'] = labelencoder.fit_transform(data['car_age'])

col1 = ['register_time', 'emission', 'year_check', 'strong_risk', 'type']
df = data.drop(columns=col1)  # 删除无效字段

# 准备训练集和测试集的特征值、目标值
target = df['car_price']
feature_name = df[['brand_name', 'car_sys', 'displacement', 'meter_mile',
                     'sell_times', 'gearbox', 'car_age', 'risk_27_check']]
feature = pd.get_dummies(feature_name)  # 哑变量编码
X_train, X_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.3)  # 直接划分训练集


# 建立模型
regressor = svm.SVR()
regressor = regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
print(score)
# print(df.head())
