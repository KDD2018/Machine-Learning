#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
from selectMySQL import SelectMySQL
from datetime import datetime
from processing import Processing
import numpy as np
import joblib as job



def get_car_info_from_customer():
    '''
    获取用户提供的车辆信息
    :return: 用户车辆表 
    '''

    brand_name = input('请输入品牌：') or '宝马'
    car_system = input('请输入车系：') or '宝马M3'
    car_model_name = input('请输入车型：') or '2018款 M3四门轿车'
    register_time = datetime.strptime(input('请输入上牌时间：') or '2019-01-01', '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（公里）：') or 10000)
    sell_times = float(input('请输入过户次数：') or 5)
    car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                              'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

    return car_info_from_customer


def get_customer_car_df(sql_to_customer_carConfig, sql_to_class, sql_to_system, sql_to_level):
    '''
    获取用户车辆的全部信息
    :param sql_to_customer_carConfig: 查询用户车辆参数配置的SQL
    :param sql_to_brand: 查询品牌集
    :param sql_to_brand：查询车系集
    :return: 用户车辆全部信息表
    '''
    # 获取客户输入车辆信息
    car_info_from_customer = get_car_info_from_customer()

    if car_info_from_customer['meter_mile'] > 550000:
        print('\n客观，您的爱车接近报废啦。。。。。')
    else:
        car_brand = car_info_from_customer['car_brand']

        # 查询客户车辆参数配置、类型
        select = SelectMySQL(host='192.168.0.3', user='clean', passwd='Zlpg1234!', db='valuation_web')
        # 根据车辆品牌、车系、车型获取用户车辆参数配置信息
        customer_carConfig = select.get_df(sql_to_customer_carConfig.format(car_brand,
                                                                            car_info_from_customer['car_system'],
                                                                            car_info_from_customer['car_model']))
        customer_car_df = car_class = classes = levels = systems = None
        if customer_carConfig:
            # 合并客户车辆配置信息和用户输入信息
            customer_car_info = dict(car_info_from_customer, **customer_carConfig[0])
            customer_car_info['price'] = 1  # 默认给出客户汽车价格为1元，方便与案例共用一套特征处理的方法
            # 客户汽车所属类别
            car_class = customer_car_info['car_class']
            # print(f'\n客官，您的爱车类型属于：{car_class}\n')

            if car_class not in ['saloon', 'suv', 'mpv', 'supercar', 'minibus', 'EV']:
                print('\n客官，现不支持此类车型的估值...')
            else:
                # 查询所有品牌、车系、车辆级别
                car_level = select.get_df(sql_to_level.format(car_brand))
                car_class = select.get_df(sql_to_class.format(car_brand))
                car_system = select.get_df(sql_to_system.format(car_brand))
                classes = [i['car_class'] for i in car_class]
                classes.remove('EV')
                systems = [i['car_system'] for i in car_system]
                levels = [i['level'] for i in car_level]

                # 将客户车辆信息写入DataFrame
                customer_car_df = pd.DataFrame([customer_car_info])
            # print(customer_car_df)
        else:
            print('\n客官，查无此车型')

        return customer_car_df, car_brand, car_class, classes, systems, levels


def load_model_predict(car_brand, x):
    '''
    加载模型进行评估
    :param path: 模型路径
    :param test_data: 测试数据集路径
    :return: 预测值
    '''
    regressor = job.load(f'../../model-param/{car_brand}.joblib')
    y_hat = regressor.predict(x)
    # your_car_price = math.expm1(y_hat)
    # print('\n**************您的爱车值这个价：%f万元**************' % your_car_price)

    return y_hat


def run():

    # 1、获取用户车辆的全部信息
    customer_car_df, car_brand, car_class, classes, systems, levels = get_customer_car_df(sql_to_customer_carConfig,
                                                                                          sql_to_class,
                                                                                          sql_to_system, sql_to_level)
    if car_class:
        # 2、对用户车信息进行处理
        process = Processing()
        categories, col_categ = process.get_category(car_class, systems, levels, classes)
        df_preprocessed = process.preprocess(customer_car_df)  # 预处理
        df_disrete = process.feature_encode(df_preprocessed, col_NEV, col_EV)  # 离散化

        df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # one-hot编码

        scale_mean = pd.read_csv('../../model-param/scale_mean_.csv').T
        scale_std = pd.read_csv('../../model-param/scale_std.csv').T
        df_numeric = df_preprocessed[['meter_mile', 'vendor_guide_price', 'price']]
        # df_numeric.ix[0, 'meter_mile'] = (df_numeric.iloc[0,0] - scale_mean.iloc[0,0]) / scale_std.iloc[0,0]
        df_numeric.ix[0, 'vendor_guide_price'] = (df_numeric.iloc[0, 1] - scale_mean.iloc[0, 1]) / scale_std.iloc[0, 1]

        df = pd.concat([df_categ, df_numeric.iloc[:, :-1]], axis=1)
        df = df.astype('float32')
        # print(df)

        # 预测用户车辆价值
        y_hat = load_model_predict(car_brand, x=df)
        y_hat = y_hat * scale_std.iloc[0,-1] + scale_mean.iloc[0, -1]
        customer_car_price = np.expm1(y_hat)
        print(f'\n客官，您的爱车值这个价：{customer_car_price}万元')



# SQL查询语句
sql_to_customer_carConfig = """
SELECT
    displacement,
	# cylinder_number,
	driving,
	gearbox_type,
	intake_form,
	maximum_power,
	voyage_range,
	car_class,
	level,
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
	# n.car_brand, 
	n.car_system,
	n.car_model,
	n.displacement,
	# n.cylinder_number,
	n.driving,
	n.gearbox_type,
	n.intake_form,
	n.maximum_power,
	n.voyage_range,
	n.car_class,
	n.level,
	n.vendor_guide_price,
	n.model_year,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_brand = '{0}'
	AND
	    n.car_class != 'EV'
	AND 
	    (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))

"""

sql_to_case_num = """
SELECT
    count(*)
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_brand = '{0}'	
"""

sql_to_class = """
select
	distinct n.car_class
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_brand = '{0}'
order by n.id
"""

sql_to_system = """
select
	distinct n.car_system
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_brand = '{0}'
order by n.id
"""

sql_to_level = """
select
	distinct n.level
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_brand = '{0}'
order by n.id
"""

# 非纯电动特征名称
col_NEV = ['car_system', 'level', 'car_class', 'displacement', 'driving', 'gearbox_type', 'intake_form',
           'maximum_power', 'register_time', 'meter_mile', 'sell_times', 'vendor_guide_price',
           'model_year', 'price']
# 纯电动特征名称
col_EV = ['car_system', 'driving', 'gearbox_type','maximum_power', 'voyage_range', 'register_time',
          'meter_mile', 'sell_times', 'vendor_guide_price', 'model_year', 'price']

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    run()