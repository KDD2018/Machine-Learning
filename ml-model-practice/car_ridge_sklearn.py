#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
import joblib as job

# pd.set_option('display.max_columns', None)


def my_car():
    '''
    获取客户车辆信息
    :return: 用户车辆表
    '''
    brand_name = input('请输入品牌：')
    car_system = input('请输入车系：')
    car_model_name = input('请输入车型：')
    register_time = datetime.strptime(input('请输入上牌时间：'), '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（公里）：'))
    gearbox = input('请输入变速箱类型：')
    displacement = float(input('请输入排量(L)：'))
    sell_times = float(input('请输入过户次数：'))
    annual_inspect = datetime.strptime(input('请输入年检到期时间：'), '%Y-%m-%d')
    # compulsory_insurance = datetime.strptime(input('请输入交强险的到期时间：'), '%Y-%m-%d')

    car_info = {'brand_name': brand_name, 'car_system': car_system, 'car_model_name': car_model_name,
                'register_time': register_time, 'meter_mile': meter_mile, 'gearbox': gearbox,
                'displacement': displacement, 'sell_times': sell_times, 'annual_inspect': annual_inspect,
                'car_price': 0}
    my_car_info = pd.DataFrame([car_info])

    return my_car_info


def conn_mysql(sql):
    '''
    sql 查询的SQL语句
    col 字段名称
    :return: data
    '''
    conn = pymysql.Connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!',
                           db='valuation_web', port=3306, charset='utf8')
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    conn.close()

    return data


def preprocess(data):
    '''
    数据预处理
    :param data: 要处理的数据框
    :return: 处理后的数据框
    '''

    data.loc[:, 'meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
    data.loc[:, 'car_price'] = data['car_price'] / 10000.0  # 换算成万元
    data = data.loc[data.meter_mile < 40, :].copy()  # 过滤掉50万公里以上的案例
    data.loc[:, 'meter_future_rate'] = 1 - data['meter_mile'] / 60  # 转换成行驶里程成新率
    data = data.dropna()
    # print(data.isnull().any())
    data.loc[:, 'annual_inspect'] = data['annual_inspect'].map(
        lambda x: ((x.year - datetime.now().year) * 12 + (x.month - datetime.now().month)) / 12)
    data.loc[:, 'car_age'] = data['register_time'].map(
        lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
    data.loc[:, 'car_price'] = data['car_price'].map(lambda x: np.log1p(x))

    return data


def feature_encode(data):
    '''
    特征离散编码
    :param data: 数据框
    :return: 特征编码后的数据框
    '''

    data.loc[:, 'displacement'] = pd.cut(data.displacement, bins=[-1, 0, 1.0, 1.6, 2.5, 4, 6, 8],
                                  labels=['0L', '0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上'])  # 排量
    data.loc[:, 'car_age'] = pd.cut(data.car_age, bins=[-1, 1, 3, 5, 8, 50],
                             labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])  # 车龄
    data.loc[:, 'sell_times'] = pd.cut(data.sell_times, bins=[-1, 0.01, 1, 2, 3, 4, 10],
                                labels=['0次', '1次', '2次', '3次', '4次', '5次及以上'])  # 过户次数
    # data.loc[:, 'annual_inspect'] = pd.cut(data.annual_inspect, bins=[-20, -10, -5, 0, 5, 20],
    #                                       labels=['10年以前', '5年以前', '过去5年', '未来5年', '未来5-10年'])

    return data


def split_data(data):
    '''
    将数据划分为训练集和测试集
    :param data: 数据框
    :return: X_train, X_test, y_train, y_test
    '''

    target = data.iloc[:,-1][:-1]
    feature_X = data.iloc[:,:-1]
    feature = pd.get_dummies(feature_X)  # 哑变量编码
    my_car_info = pd.DataFrame([feature.iloc[-1, :].copy()])
    feature = feature.iloc[:-1, :]
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

    return  X_train, X_test, y_train, y_test, my_car_info, feature, target


def model_and_persist(feature, target, car_class):
    '''
    确立最优默型 
    :param X_train: 训练特征
    :param X_test: 测试特征
    :param y_train: 训练目标
    :param y_test: 测试目标
    :param my_car_info: 用户的车辆信息
    '''

    print('\n**************开始训练Ridge模型**************')
    regressor = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10], store_cv_values=True)

    # regressor = regressor.fit(X_train, y_train)
    # score = regressor.score(X_test, y_test)
    regressor = regressor.fit(feature, target)
    score = regressor.score(feature, target)
    print('\n最优超参数alpha：%f'%regressor.alpha_)  # 最优alpha
    print('\n**************拟合优度为：%f******************'%score)

    prediction = regressor.predict(X_test)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction, multioutput='uniform_average')
    print('\n**************均方误差为：%f**************'%mse)
    error = prediction - y_test
    abe = sum(abs(error)) / len(y_test)
    print('\n**************平均绝对误差为：%f**************'%abe)

    job.dump(regressor, './model-param/{0}.joblib'.format(car_class))


def prediction(my_car_info, car_class):
    '''
    预测客户车辆价格
    :param my_car_info: 客户车辆信息
    :return: 
    '''
    regressor = job.load('./model-param/{0}.joblib'.format(car_class))
    log_price = regressor.predict(my_car_info)
    your_car_price = math.expm1(log_price)
    print('\n**************您的爱车值这个价：%f**************' %your_car_price)


# SQL查询语句
sql_to_case = """
select
	brand_name,
	vehicle_system_name,
	car_model_name,
	register_time,
	meter_mile,
	semiautomatic_gearbox,
	displacement,	
	sell_times,
	year_check_end_time,
	# model_year,
	price
from
	second_car_sell
where
    car_class="{0}"
and
    (risk_27_check = "0"
or 
    risk_27_check is null)
"""


sql_to_class = """
select
	car_class
from
	second_car_sell
where
    brand_name = "{0}"
and
    vehicle_system_name = "{1}"
and
	car_model_name = "{2}"
"""

col = ['brand_name', 'car_system', 'car_model_name', 'register_time', 'meter_mile', 'gearbox', 'displacement',
       'sell_times', 'annual_inspect', 'car_price']

col1 = ['brand_name', 'car_system', 'car_age', 'meter_future_rate', 'gearbox', 'displacement', 'sell_times',
        'annual_inspect', 'car_price']

car_class_dict = {'suv': ['大型SUV', '中大型SUV', '小型SUV', '紧凑型SUV', '中型SUV'],
                  'saloon': ['小型车', '大型车', '微型车', '中大型车', '中型车', '紧凑型车'], 'supercar': '跑车',
                  'pickup': ['皮卡', '微卡', '高端皮卡'], 'mpv': 'MPV', 'minibus':['轻客', '微面']}


if __name__ == '__main__':

    # 获取用户车辆信息
    my_car_info = my_car()

    start_time  = datetime.now()

    # 获取MySQL数据
    car_level = conn_mysql(sql_to_class.format(my_car_info.loc[0,'brand_name'],
                                               my_car_info.loc[0,'car_system'],
                                               my_car_info.loc[0, 'car_model_name']))
    car_class = car_level[0][0]
    print('\n您的爱车类型属于：%s'%car_class)

    data = conn_mysql(sql_to_case.format(car_level[0][0]))
    data = pd.DataFrame(list(data), columns=col)
    data = pd.concat([data, my_car_info], sort=False, ignore_index=True)

    # 数据预处理
    data = preprocess(data)

    # 特征编码
    data = feature_encode(data)

    # 有效字段
    df = data.loc[:, col1].copy()
    df.index = range(len(df))

    # 划分数据集
    X_train, X_test, y_train, y_test, my_car_info, feature, target = split_data(df)

    # 建立Ridge回归模型及其持久化
    # model_and_persist(feature, target, car_class)

    # 预测客户爱车价格
    price = prediction(my_car_info, car_class)
    end_time = datetime.now()
    sec = (end_time-start_time).seconds
    print('\n运行时间：%.2f 秒'%sec)
