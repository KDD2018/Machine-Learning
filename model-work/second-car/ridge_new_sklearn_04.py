#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import RidgeCV
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import joblib as job
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)


def get_customer_car():
    '''
    获取预估车辆信息
    :return: 用户车辆表
    '''

    brand_name = input('请输入品牌：') or '雪铁龙'
    car_system = input('请输入车系：') or '雪铁龙C3-XR'
    car_model_name = input('请输入车型：') or '2015款 1.6THP 自动旗舰型'
    register_time = datetime.strptime(input('请输入上牌时间：') or '2016-09-01', '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（公里）：') or 45500)
    if meter_mile < 550000:
        sell_times = float(input('请输入过户次数：') or 0)
        # annual_inspect = datetime.strptime(input('请输入年检到期时间：') or '2020-09-01', '%Y-%m-%d')
        customer_car_info = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                    'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}
                    # 'year_check_end_time': annual_inspect}
    else:
        print('客官，您的爱车已近报废了。。。。。。')

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
                           cursorclass=pymysql.cursors.SSDictCursor,
                           connect_timeout=7200)
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
    data = data.loc[data.meter_mile < 55, :].copy()  # 过滤掉40万公里以上的案例
    data.loc[:, 'meter_mile'] = 1 - data['meter_mile'] / 60  # 转换成行驶里程成新率
    data.loc[:, 'price'] = data['price'] / 10000.0  # 换算成万元
    data.loc[:, 'vendor_guide_price'] = data['vendor_guide_price'] / 10000.0  # 换算成万元
    # data.loc[:, 'year_check_end_time'] = data['year_check_end_time'].map(
    #     lambda x: ((x.year - datetime.now().year) * 12 + (x.month - datetime.now().month)) / 12)
    data.loc[:, 'register_time'] = data['register_time'].map(
        lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
    data.loc[:, 'vendor_guide_price'] = data['vendor_guide_price'].map(lambda x: np.log1p(x))
    data.loc[:, 'price'] = data['price'].map(lambda x: np.log1p(x))

    return data


def feature_encode(data, col1, col2):
    '''
    特征离散编码
    :param data: 数据框
    :return: 离散化处理的数据框
    '''
    if data.loc[0, 'car_class'] != 'EV':
        data = data.loc[:, col1]
        # 气缸离散化
        data.loc[:, 'cylinder_number'] = pd.cut(data.cylinder_number, bins=[0, 2, 3, 4, 5, 6, 8, 16],
                                                labels=['2缸', '3缸', '4缸','5缸', '6缸', '8缸', '10缸以上'])
        # # 排量离散化
        # data.loc[:, 'displacement'] = pd.cut(data.displacement, bins=[0, 1.0, 1.6, 2.5, 4, 6, 8],
        #                                      labels=['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上'])
    else:
        data = data.loc[:, col2]
    data = data.dropna()
    data.index = range(len(data))


    # 最大功率离散化
    data.loc[:, 'maximum_power'] = pd.cut(data.maximum_power, bins=[ 0, 100, 150, 200, 250, 500, 1000],
                                          labels=['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上'])
    # 上牌时间
    data.loc[:, 'register_time'] = pd.cut(data.register_time, bins=[-1, 1, 3, 5, 8, 50],
                                          labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])
    # 过户次数
    data.loc[:, 'sell_times'] = pd.cut(data.sell_times, bins=[-1, 0.01, 1, 2, 3, 4, 10],
                                       labels=['0次', '1次', '2次', '3次', '4次', '5次及以上'])
    # # 年检到期
    # data.loc[:, 'year_check_end_time'] = pd.cut(data.year_check_end_time, bins=[-20, -10, -5, 0, 5, 20],
    #                                             labels=['10年以前', '5年以前', '过去5年', '未来5年', '未来5-10年'])
    # 车款年份
    data.loc[:, 'model_year'] = pd.cut(data.model_year, bins=[0, 2008, 2013, 2017, 2050],
                                       labels=['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后'])
    # 车况
    data.loc[:, 'vehicle_condition'] = pd.cut(data.car_loss, bins=[-1, 0, 8, 16, 24, 100],
                                              labels=['车况优秀', '车况良好', '车况一般', '车况较差', '车况极差'])


    return data


def split_data(data):
    '''
    将数据划分为训练集和测试集
    :param data: 数据框
    :return: X_train, X_test, y_train, y_test
    '''

    target = data.iloc[:,-1]  # 目标值
    feature = data.iloc[:,:-1]  # 特征值
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

    return  X_train, X_test, y_train, y_test, feature, target


def onehot_encode(data, categories):
    '''
    One-Hot编码
    :param data: 待编码的分类特征
    :return: One-Hot编码后的数据
    '''

    enc = OneHotEncoder(sparse=False, categories=categories)
    data_encode = enc.fit_transform(data)
    df = pd.DataFrame(data_encode, columns=enc.get_feature_names())

    return df


def modeling_and_persist(feature, target, customer_car_info):
    '''
    确立最优模型 
    :param X_train: 训练特征
    :param X_test: 测试特征
    :param y_train: 训练目标
    :param y_test: 测试目标
    :param my_car_info: 用户的车辆信息
    '''
    print('\n**************开始训练Ridge模型**************')
    cv = ShuffleSplit(n_splits=10, test_size=0.25)
    regressor = RidgeCV(alphas=[0.01,0.1,1], cv=cv)

    # regressor = regressor.fit(X_train, y_train)
    # score = regressor.score(X_test, y_test)
    regressor = regressor.fit(feature, target)
    score = regressor.score(feature, target)
    # print('\n最优超参数alpha：%f'%regressor.alpha_)  # 最优alpha
    print('\n**************拟合优度为：%.4f******************'%score)

    # prediction = regressor.predict(X_test)
    # mse = mean_squared_error(y_true=y_test, y_pred=prediction, multioutput='uniform_average')
    # print('\n**************均方误差为：%f**************'%mse)
    # error = prediction - y_test
    # abe_true = abs(error).map(lambda x: math.expm1(x))
    # abe =  sum(abe_true)/ len(y_test)
    # print('\n**************平均绝对误差为：%f**************'%abe)

    job.dump(regressor, '../model-param/{0}.joblib'.format(customer_car_info['car_class']))


def predict(my_car_df, customer_car_info):
    '''
    预测车价
    :param my_car_df: 待估车辆配置信息
    :return: 车价
    '''
    regressor = job.load('../model-param/{0}.joblib'.format(customer_car_info['car_class']))
    y_hat = regressor.predict(my_car_df)
    your_car_price = math.expm1(y_hat)
    print('\n**************您的爱车值这个价：%f万元**************' %your_car_price)


# SQL查询语句
sql_to_customer_carConfig = """
SELECT
	cylinder_number,
	driving,
	gearbox_type,
	intake_form,
	maximum_power,
	voyage_range,
	car_class,
	vendor_guide_price,
	model_year
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
	n.driving,
	n.gearbox_type,
	n.intake_form,
	n.maximum_power,
	n.voyage_range,
	n.car_class,
	n.vendor_guide_price,
	n.model_year,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	c.car_loss,
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
	INNER JOIN second_car_check2 c ON c.second_car_sell_id = s.id
WHERE
	n.car_class = '{0}'
	AND 
	    n.model_year >= '{1}'
"""

sql_to_brand_and_system = """
select
	car_brand,
	car_system
from
	new_car_information
where
	car_class = '{0}'
	and
	    model_year >= '{1}'
order by id
"""



# 非纯电动特征名称
col_NEV = ['car_brand', 'car_system', 'cylinder_number', 'driving', 'gearbox_type', 'intake_form',
           'maximum_power', 'register_time', 'meter_mile', 'sell_times', 'vendor_guide_price',
           'model_year', 'car_loss', 'price']

# 纯电动特征名称
col_EV = ['car_brand', 'car_system', 'driving', 'gearbox_type','maximum_power', 'voyage_range', 'register_time',
          'meter_mile', 'sell_times', 'vendor_guide_price', 'model_year', 'car_loss', 'price']


# 分类型特征名称
col_categories_NEV = ['car_brand', 'car_system', 'cylinder_number', 'driving', 'gearbox_type',
                      'intake_form', 'maximum_power', 'register_time', 'sell_times', 'model_year', 'vehicle_condition']
col_categories_EV = ['car_brand', 'car_system', 'driving', 'gearbox_type', 'maximum_power', 'register_time',
                     'sell_times', 'model_year', 'vehicle_condition']


# 用于过滤太老旧的车
model_year_dict = {'saloon': 2005, 'suv': 2007, 'mpv': 2006, 'minibus': 2007, 'supercar': 0, 'EV': 0}

# 分类型特征类别
cylinder_number = ['2缸', '3缸', '4缸','5缸', '6缸', '8缸', '10缸以上']
# displacement = ['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上']
driving = ['四驱', '后驱', '前驱']
gearbox_type = ['DCT', 'CVT', 'AMT', 'AT自动', 'MT手动', '超跑变速箱', '单速变速箱']
intake_form = ['自然吸气', '双增压', '涡轮增压', '机械增压', '四涡轮增压']
maximum_power = ['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上']
register_time = ['1年以内', '1-3年', '3-5年', '5-8年', '8年以上']
sell_times = ['0次', '1次', '2次', '3次', '4次', '5次及以上']
# year_check_end_time = ['10年以前', '5年以前', '过去5年', '未来5年', '未来5-10年']
model_year = ['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后']
vehicle_condition = ['车况优秀', '车况良好', '车况一般', '车况较差', '车况极差']



categories_NEV = [cylinder_number, driving, gearbox_type, intake_form, maximum_power, register_time,
                  sell_times, model_year, vehicle_condition]
categories_EV = [driving, gearbox_type, maximum_power, register_time, sell_times, model_year, vehicle_condition]



if __name__ == '__main__':

    # 1、获取用户车辆品牌、车系、车型、上牌时间、行驶里程、过户次数、年检到期时间等信息
    customer_car = get_customer_car()
    # print(customer_car)
    start_time = datetime.now()


    # 2、查询客户车辆参数配置、类型
    customer_carConfig = conn_mysql(sql_to_customer_carConfig.format(customer_car['car_brand'],
                                                                     customer_car['car_system'],
                                                                     customer_car['car_model']))
    # print(customer_carConfig)
    if not customer_carConfig:
        print('\n客官，查无此车型')
    else:
        customer_car_info = dict(customer_car, **customer_carConfig[0])  # 合并客户车辆配置信息和使用情况
        customer_car_info['price'] = 1  # 默认给出客户汽车价格为1元，方便与案例共用一套特征处理的方法
        car_class = customer_car_info['car_class']  # 客户汽车所属类别
        print('\n客官，您的爱车类型属于：%s\n'%car_class)

        if car_class not in ['saloon', 'suv', 'mpv', 'supercar', 'minibus', 'EV']:
            print('客官，现不支持此类车型的估值')
        else:
            customer_car_df = pd.DataFrame([customer_car_info])  # 将客户车辆信息写入DataFrame
            # print(customer_car_df)

            # 3、查询同类型车的品牌集、车系集
            car_class_config = conn_mysql(sql_to_brand_and_system.format(car_class, model_year_dict[car_class]))
            # print(car_class_config)
            car_class_config = pd.DataFrame(car_class_config)
            car_brand = sorted(list(set(car_class_config['car_brand'].values)))  # 品牌集
            car_system = sorted(list(set(car_class_config['car_system'].values)))  # 车系集


            if car_class == 'EV':
                categories_EV.insert(0, car_brand)
                categories_EV.insert(1, car_system)
                categories = categories_EV
                col_categ = col_categories_EV
            else:
                categories_NEV.insert(0, car_brand)
                categories_NEV.insert(1, car_system)
                categories = categories_NEV
                col_categ = col_categories_NEV


            ##################################### 训练模型 #############################################
            # 4、二手车案例信息
            car_case= conn_mysql(sql_to_CarConfig_CarCase.format(car_class, model_year_dict[car_class]))
            car_case_df = pd.DataFrame(car_case)  # 将同类车辆案例信息写入DataFrame


            # 5、案例数据处理
            data = preprocess(car_case_df)  # 数据变换
            data = feature_encode(data, col_NEV, col_EV)  # 离散化处理
            print(data.shape)

            # # data = data.loc[data.car_system == '哈弗H6']
            # # 统计分析
            # # cormat = data.corr()
            # plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
            # # sns.distplot(data['preservation'])
            # # figure, ax = plt.subplots()
            # sns.boxplot(data['vehicle_condition'], data['price'])
            # # sns.scatterplot(data['car_loss'], data['price'])
            # # stats.probplot(data['preservation'], plot=plt) # Q-Q图
            # # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # # sns.heatmap(cormat,square=True)
            # plt.show()

            df_categories = onehot_encode(data[col_categ], categories=categories)  # One-Hot编码
            df = pd.concat([df_categories, data[['meter_mile', 'vendor_guide_price', 'price']]], axis=1)
            print(df.shape)

            # 6、划分数据集
            X_train, X_test, y_train, y_test, feature, target = split_data(df)

            # 7、建立机器学习模型
            modeling_and_persist(feature, target, customer_car_info)



            ###################################### 预测 ##################################################
            # # 4、客户车辆信息处理
            # customer_car_df = preprocess(customer_car_df)  # 数据变换
            # customer_car_df = feature_encode(customer_car_df, col_NEV, col_EV)  # 离散化处理
            # my_car_df_encode = onehot_encode(customer_car_df[col_categ], categories=categories)  # One-Hot编码
            # my_car_df = pd.concat([my_car_df_encode, customer_car_df[['meter_mile', 'vendor_guide_price']]], axis=1)
            #
            #
            # # 预测客户汽车价格
            # predict(my_car_df, customer_car_info)


    end_time = datetime.now()
    sec = (end_time - start_time).seconds
    print('\n运行时间：%.2f 秒' % sec)






