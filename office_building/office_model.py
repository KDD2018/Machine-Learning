#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymongo
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', None)


def connect(sql):
    conn = pymysql.Connect(host='192.168.0.3', user='pnggu', passwd='Pinggu123', db='model', port=3306, charset='utf8')
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    conn.close()
    data = pd.DataFrame(list(data))
    return data

def floor_standard(current_floor, total_floor):
    floor_at = [i for i in current_floor]
    for index, item in enumerate(list(zip(current_floor, total_floor))):
        if item[0] in ['低区','中区','高区']:
            floor_at[index] = item[0]
        else:
            if int(item[0]) / item[1] <= 1.0/3.0:
                floor_at[index] = '低区'
            elif int(item[0]) / item[1] > 2.0/3.0:
                floor_at[index] = '高区'
            else:
                floor_at[index] = '中区'
    return floor_at



# SQL语句
sql = """
select
	area_name,
	building_name,
	building_support_count_new,
	left(completion_time_new, 4),
	current_floor,
	district_full_name,
	number_of_freight_elevators,
	number_of_parking_space_on_the_ground,
	number_of_passenger_elevators_new,
	number_of_underground_parking_space_new,
	office_type_new,
	property_fee,
	property_grade,
	renovation,
	renovation_new,
	residential_area,
	selling_time,
	standard_floor_area,
	total_floor,
	total_price,
	unit_price
from
	office_handle_2
"""

# 1. 获取特征数据
col = ['area_name','building_name','building_support','completion_time','current_floor','district','freight_elevators',
       'parking_ground','passenger_elevators','underground_parking','office_type','property_fee','office_grade',
       'renovation','segmention','residential_area','selling_time','standard_floor_area','total_floor',
       'total_price','unit_price']
df_office = connect(sql)
df_office.columns = col
df_office.to_csv('/home/kdd/Desktop/office.csv', header=True, encoding="utf_8_sig")  # 写入从csv文件
# print(df_office.shape)


# 2. 预处理（删除异常值和缺失值）


df_office = df_office.drop(index=df_office[df_office.completion_time==2020].index)
df_office = df_office.drop(index=df_office[df_office.renovation==''].index)
df_office = df_office.drop(index=df_office[df_office.current_floor.isnull()].index)
df_office = df_office.drop(index=df_office[df_office.building_name=='投资广场'].index)
df_office = df_office.drop(index=df_office[df_office.property_fee.isnull()].index)
df_office = df_office.drop(index=df_office[df_office.passenger_elevators.isnull()].index)
df_office = df_office.drop(index=df_office[df_office.underground_parking.isnull()].index)
df_office = df_office.drop(index=df_office[df_office.standard_floor_area=='1350㎡09出售'].index)
df_office = df_office.drop(index=df_office[df_office.standard_floor_area=='2000 ㎡\xa0 总建筑'].index)
df_office = df_office.drop(index=df_office[df_office.standard_floor_area.isnull()].index)
df_office = df_office.drop(index=df_office[df_office.property_fee==900].index)
df_office = df_office.drop(index=df_office[df_office.property_fee==200].index)


df_office['completion_time'] = df_office['completion_time'].astype(int)  # 转换建成年代为int类型
df_office['property_fee'] = df_office['property_fee'].astype(float)
df_office['standard_floor_area'] = df_office['standard_floor_area'].astype(float)
df_office['passenger_elevators'] = df_office['passenger_elevators'].astype(int)
df_office['underground_parking'] = df_office['underground_parking'].astype(int)
df_office['unit_price'] = df_office['unit_price'].astype(float)
df_office['total_floor'] = df_office['total_floor'].astype(int)

df_office = df_office.drop(index=df_office[df_office.unit_price>100000].index)
df_office = df_office.drop(index=df_office[df_office.unit_price<10000].index)
df_office['floor_at'] = floor_standard(df_office.current_floor, df_office.total_floor)
# 3. 特征工程（离散编码）
df_office['support'] = pd.cut(df_office.building_support, bins=[-1,1,7,9], labels=[1,2,3])  # 楼内配套
df_office['office_age'] = pd.cut(df_office.completion_time, bins=[0,2000,2005,2010,2020], labels=[0,1,2,3])  # 写字楼房龄
df_office['property'] = pd.cut(df_office.property_fee, bins=[0,20,40,100], labels=[1,2,3])  # 物业费
df_office['standard_area'] = pd.cut(df_office.standard_floor_area, bins=[0,2000,6000,10000,20000], labels=[1,2,3,4])  # 标准层面积
df_office['passenger_ele'] = pd.cut(df_office.passenger_elevators, bins=[0,5,10,15,25,50], labels=[0,1,2,3,4])  # 电梯
df_office['parking'] = pd.cut(df_office.underground_parking, bins=[0,500,1000,2000,5000], labels=[0,1,2,3])  # 停车位
# print(df_office.isnull().any()) # 查看字段是否缺失数据
df_office['decoration'] = df_office['renovation'].map({'毛坯': 0, '简装修': 1, '精装修': 2})
df_office['floor_at'] = df_office['floor_at'].map({'低区':0, '中区':1, '高区':2})
df_office['segment'] = df_office['segmention'].map({'不可分割':0, '可分割':1})
df_office['district'] = df_office['district'].map({'东城':1, '西城':2, '朝阳':3, '海淀':4, '通州':5, '丰台':6, '大兴':7,
                                                   '昌平':8, '顺义':9})
# label_mapping = {lab:idx for idx,lab in enumerate(set(df_office['area_name']))}
# df_office['area_name'] = df_office['area_name'].map(label_mapping)  # 商圈编码
# print(set(df_office.district))



# 按条件筛选至某一商圈内的数据
df = df_office[df_office['office_type']=='纯写字楼']
df = df[df['office_grade']=='甲级']
df = df[df['district']==3]  # 朝阳区
df = df[df['area_name']=='CBD']
# df = df[df['property']==2]
# df = df[df['office_age']==3]
# # df = df[df['decoration']==2]
# # df =df[df['segment']==1]
# # print(set(df['property_fee']))


# # 条形图
# pr = df['unit_price'].groupby(by=df.decoration).mean()
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# # plt.scatter(df.property_fee, df.unit_price)
# plt.bar(pr.index, pr, width=0.3, color='g')
# plt.show()



# 获取样本数据
df = df[[ 'office_age', 'floor_at', 'passenger_ele','parking', 'office_type', 'property',
          'decoration', 'segment', 'standard_area', 'unit_price']]
print(df.shape)

target = df['unit_price']  # 目标值
del df['unit_price'], df['office_type']
feature = df  # 特征值



# 4. 建立模型
# 决策树
# regressor = tree.DecisionTreeRegressor(max_depth=7)
# 随机森林
regressor = RandomForestRegressor(max_depth=7)
# X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.4)  # 直接划分训练集
# regressor = regressor.fit(X_train, y_train)
# score = regressor.score(X_test, y_test)

# K-折交叉验证
scores = cross_val_score(regressor,feature, target, cv=4)
print(scores.mean())



