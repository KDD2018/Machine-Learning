#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
import tensorflow as tf

pd.set_option('display.max_columns', None)



def get_customer_car():
    '''
    获取预估车辆信息
    :return: 用户车辆表
    '''

    brand_name = input('请输入品牌：') or '大众'
    car_system = input('请输入车系：') or '桑塔纳'
    car_model_name = input('请输入车型：') or '2016款 1.6L 自动风尚版'
    register_time = datetime.strptime(input('请输入上牌时间：') or '2018-01-01', '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（万公里）：') or 1.71)
    sell_times = float(input('请输入过户次数：') or 0)
    annual_inspect = datetime.strptime(input('请输入年检到期时间：') or '2020-01-01', '%Y-%m-%d')

    customer_car_info = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times,
                'year_check_end_time': annual_inspect}
    # my_car_info = pd.DataFrame([car_info])

    return customer_car_info


def conn_mysql(sql):
    '''
    sql 查询的SQL语句
    col 字段名称
    :return: data
    '''
    conn = pymysql.Connect(host='192.168.0.3',
                           user='clean',
                           passwd='Zlpg1234!',
                           db='valuation_web',
                           port=3306,
                           charset='utf8',
                           cursorclass=pymysql.cursors.SSDictCursor, connect_timeout=7200)
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

    if data.loc[0, 'car_class'] != 'EV':
        del data['voyage_range']
    data.loc[:, 'meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
    data = data.loc[data.meter_mile < 40, :].copy()  # 过滤掉40万公里以上的案例
    data.loc[:, 'meter_mile'] = 1 - data['meter_mile'] / 60  # 转换成行驶里程成新率
    data.loc[:, 'preservation'] = data['price'] / data['vendor_guide_price'] # 保值率
    data.loc[:, 'vendor_guide_price'] = data['vendor_guide_price'] / 10000.0

    # print(data.isnull().any())
    data.loc[:, 'year_check_end_time'] = data['year_check_end_time'].map(
        lambda x: ((x.year - datetime.now().year) * 12 + (x.month - datetime.now().month)) / 12)
    data.loc[:, 'register_time'] = data['register_time'].map(
        lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
    # data.loc[:, 'preservation'] = data['preservation'].map(lambda x: np.log1p(x))
    del data['car_model'], data['car_class'], data['price']
    # print(data.shape)
    data = data.dropna()
    # print(data.shape)

    return data


def feature_encode(data):
    '''
    特征离散编码
    :param data: 数据框
    :return: 特征编码后的数据框
    '''

    # 气缸离散化
    data.loc[:, 'cylinder_number'] = pd.cut(data.cylinder_number, bins=[0, 5, 6, 8, 16],
                                         labels=['5缸以内', '6缸', '8缸', '8缸以上'])
    # 排量离散化
    data.loc[:, 'displacement'] = pd.cut(data.displacement, bins=[-1, 0, 1.0, 1.6, 2.5, 4, 6, 8],
                                         labels=['0L', '0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上'])
    # 上牌时间
    data.loc[:, 'register_time'] = pd.cut(data.register_time, bins=[-1, 1, 3, 5, 8, 50],
                                          labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])
    # 过户次数
    data.loc[:, 'sell_times'] = pd.cut(data.sell_times, bins=[-1, 0.01, 1, 2, 3, 4, 10],
                                       labels=['0次', '1次', '2次', '3次', '4次', '5次及以上'])
    # 年检到期
    data.loc[:, 'year_check_end_time'] = pd.cut(data.year_check_end_time, bins=[-20, -10, -5, 0, 5, 20],
                                           labels=['10年以前', '5年以前', '过去5年', '未来5年', '未来5-10年'])
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


def model_select_and_predict(my_car_info, feature, target):
    '''
    确立最优默型 
    :param X_train: 训练特征
    :param X_test: 测试特征
    :param y_train: 训练目标
    :param y_test: 测试目标
    :param my_car_info: 用户的车辆信息
    '''
    print('\n**************开始训练Ridge模型**************')
    # cv = ShuffleSplit(n_splits=10, test_size=0.25)
    regressor = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10], store_cv_values=True)

    # regressor = regressor.fit(X_train, y_train)
    # score = regressor.score(X_test, y_test)
    regressor = regressor.fit(feature, target)
    score = regressor.score(feature, target)
    print('\n最优超参数alpha：%f'%regressor.alpha_)  # 最优alpha
    print('\n**************拟合优度为：%f******************'%score)

    # prediction = regressor.predict(X_test)
    # mse = mean_squared_error(y_true=y_test, y_pred=prediction, multioutput='uniform_average')
    # print('\n**************均方误差为：%f**************'%mse)
    # error = prediction - y_test
    # abe = sum(abs(error)) / len(y_test)
    # print('\n**************平均绝对误差为：%f**************'%abe)


    log_price = regressor.predict(my_car_info)
    your_car_price = math.expm1(log_price)
    print('\n**************您的爱车值这个价：%f**************' %your_car_price)


def ridge(X_train, y_train, X_test, y_test, alpha):
    '''
    Tensorflow实现Ridge回归
    :param X: 特征值
    :param y: 目标值
    :return: Ridge模型
    '''
    # 初始化参数
    learning_rate = 0.01
    training_epochs = 500

    # 数据占位符
    X = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # 初始化权重和偏置
    W = tf.Variable(tf.random_normal(shape=[X_train.shape[1], 1]), name='Weight')
    b = tf.Variable(tf.random_normal(shape=[1,1]), name='Bias')

    # 建立模型
    y_hat = tf.add(tf.matmul(X, W), b)

    # 损失函数
    cost = tf.reduce_mean(tf.square(y_hat - y)) + alpha * tf.norm(W)

    # 梯度下降优化
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # 初始化变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(training_epochs):
            sess.run(train_op, feed_dict={X: X_train, y: y_train})
            print('第epoch次训练，损失为：%f'%sess.run(cost))
            if epoch % training_epochs == 0:
                loss = sess.run(cost, feed_dict={X: X_test, y: y_test})
                weights = sess.run(W)
                bias = sess.run(b)

    return weights, bias, loss

# SQL查询语句
sql_to_customer_carConfig = """
SELECT
	cylinder_number,
	displacement,
	driving,
	gearbox_type,
	intake_form,
	maximum_power,
	fuel_consumption,
	voyage_range,
	maximum_torque,
	vendor_guide_price,
	car_class
FROM
	new_car_information
WHERE
    car_brand = "{0}"
AND
    car_system = "{1}"
AND
	car_model = "{2}"
"""

sql_to_CarConfig_CarCase = """
SELECT
	n.car_brand, 
	n.car_system,
	n.car_model,
	n.cylinder_number,
	n.displacement,
	n.driving,
	n.gearbox_type,
	n.intake_form,
	n.maximum_power,
	n.fuel_consumption,
	n.voyage_range,
	n.maximum_torque,
	n.vendor_guide_price,
	n.car_class,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	s.year_check_end_time,
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id 
WHERE
	s.car_class = '{0}'
AND 
    n.car_class = '{1}'
AND
    (s.risk_27_check = "0"
OR
    s.risk_27_check is null)   
"""

col = ['car_brand', 'car_system', 'car_model', 'cylinder_number', 'displacement', 'driving', 'gearbox_type',
       'intake_form', 'maximum_power', 'fuel_consumption', 'voyage_range','maximum_torque','vendor_guide_price',
       'car_class', 'register_time', 'meter_mile', 'sell_times', 'year_check_end_time', 'price']

col1 = ['brand_name', 'car_system', 'car_age', 'meter_future_rate', 'gearbox', 'displacement', 'sell_times',
        'annual_inspect', 'car_price']


if __name__ == '__main__':

    # 获取用户车辆品牌、车系、车型、上牌时间、行驶里程、过户次数、年检到期时间等信息
    customer_car = get_customer_car()

    # 查询客户车辆参数配置、类型
    customer_carConfig = conn_mysql(sql_to_customer_carConfig.format(customer_car['car_brand'],
                                                                     customer_car['car_system'],
                                                                     customer_car['car_model']))
    customer_car_info = dict(customer_car, **customer_carConfig[0])  # 合并客户车辆配置信息和使用情况
    customer_car_info['price'] = 1  # 默认给出客户汽车价格为1元，方便与其他案例一起进行特征处理
    print('\n您的爱车类型属于：%s\n'%customer_car_info['car_class'])  # 打印提示所属车型类别
    customer_car_df = pd.DataFrame([customer_car_info])  # 将客户车辆信息写入DataFrame

    # 查询同类型车辆案例信息
    car_case= conn_mysql(sql_to_CarConfig_CarCase.format(customer_car_info['car_class'],
                                                         customer_car_info['car_class']))
    car_case_df = pd.DataFrame(list(car_case), columns=col)  # 将同类车辆案例信息写入DataFrame
    car_df = pd.concat([car_case_df, customer_car_df], sort=False, ignore_index=True)  # 将案例与客户车辆信息合并

    # 缺失值判断
    # print(car_df.head())
    # df = car_df[car_df['gearbox_type'].isnull()]
    # print(df.shape)
    # print(data['register_time'].dtype)
    # print(set(car_df['cylinder_number']))
    # df.to_csv('/home/kdd/python/car/saloon_displacement.csv', encoding='gbk', chunksize=10000)
    # print(data.tail())
    # print(car_df.isnull().any())

    # 数据预处理
    data = preprocess(car_df)
    # print(data.tail())

    # 特征编码
    data = feature_encode(data)

    # 统计分析
    # cormat = data.corr()
    # plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
    # sns.distplot(data['preservation'])
    # figure, ax = plt.subplots()
    # sns.boxplot(data['cylinder_number'], data['fuel_consumption'])
    # # sns.boxplot(data['compulsory_insurance'], data['car_price'])
    # # # sns.scatterplot(data['meter_mile'], data['car_price'])
    # # # stats.probplot(data['car_price'], plot=plt)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # sns.heatmap(cormat,square=True)
    # plt.show()



    # 有效字段
    df = data.loc[:, col1].copy()
    df.index = range(len(df))
    # print(df.shape)

    # 划分数据集
    # X_train, X_test, y_train, y_test, my_car_info, feature, target = split_data(df)
    # print(X_train.shape)
    # print(X_test.shape)

    # 建立机器学习模型
    # model_select_and_predict(my_car_info, feature, target)

    # Tensorflow-Ridge model
    # weights, bias, loss = ridge(X_train, y_train, X_test, y_test, alpha=0.1)
    # print(weights)
    # print(bias)
    # print(loss)





