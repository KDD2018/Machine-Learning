#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymongo
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymysql
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def connect_mongo(projection):
    client = pymongo.MongoClient('mongodb://spider:spider123@192.168.0.5:27017/')
    db = client['second_hand_car_autohome']
    collection = db['全国']
    data = pd.DataFrame(list(collection.find({}, projection)))
    return data



pd.set_option('display.max_columns', None)

# 查询文档字段
project = {'_id': 0, 'title': 1, 'car_address': 1, 'displacement': 1,'emission_standard': 1,
              'is_have_strong_risk': 1, 'level': 1, 'meter_mile': 1, 'registe_time': 1,'sell_times': 1,
              'semiautomatic_gearbox': 1, 'year_check_end_time': 1, 'car_price': 1}

# 获取Mongodb数据
data = connect_mongo(project)
# data.to_csv('/home/kdd/Desktop/car.csv', encoding='gbk')  # 写入csv

# 数据预处理
data = data.drop(index=data[data.registe_time==''].index)
data = data.drop(index=data[data.car_price==''].index)  # 删除空字符

data.displacement = data['displacement'].map(lambda x: x.split('L')[0])  # 排量
data['displacement'] = data.displacement.astype(float)
data.meter_mile = data['meter_mile'].map(lambda x: x.split('万')[0])  # 行驶里程
data['meter_mile'] = data['meter_mile'].astype(float)
data.sell_times = data['sell_times'].map(lambda x: x.split('次')[0])  # 过户次数
data['sell_times'] = data['sell_times'].replace('',np.nan)
data['sell_times'] = data['sell_times'].fillna(0)
data['sell_times'] = data.sell_times.astype(int)
data.registe_time = pd.to_datetime(data['registe_time'].map(lambda x: str(x) + '01'))  # 上牌时间
data.registe_time = data.registe_time.map(lambda x: datetime.strftime(x, '%Y'))
data['car_age'] = data['registe_time'].map(lambda x: 2019-int(x))



# 特征编码
data['displacement'] = pd.cut(data.displacement, bins=[-1,0.01,1,1.6,2,3,4,8],
                              labels=['0L','0.01-1L','1-1.6L','1.6-2L','2-3L','3-4L','4L以上'])  # 排量
data['car_age'] = pd.cut(data.car_age, bins=[-1,1.1,3.1,5.1,8.1,50],
                              labels=['1年以内','3年以内','5年以内','8年以内','8年以上'])  # 车龄
data['sell_times'] = pd.cut(data.sell_times, bins=[-1,1,3,5,8,20],
                              labels=['0次','2次以内','4次以内','7次以内','7次以上'])  # 过户次数

labelencoder = LabelEncoder() # 标签实例化
data['sell_times'] = labelencoder.fit_transform(data['sell_times'])

print(set(data.sell_times))
# print(set(data.level))
# print(set(data.life_span))
print(data.head())
# print(type(data.car_price[0]))