#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
from selectMySQL import SelectMySQL
from datetime import datetime
from processing import Processing




pd.set_option('display.max_columns', None)


def get_car_info_from_customer():
    '''
    获取用户提供的车辆信息
    :return: 用户车辆表
    '''

    brand_name = input('请输入品牌：') or '大众'
    car_system = input('请输入车系：') or '途观'
    car_model_name = input('请输入车型：') or '2016款 280TSI 自动两驱丝绸之路舒适版'
    register_time = datetime.strptime(input('请输入上牌时间：') or '2016-09-01', '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（万公里）：') or 30)
    # car_condition = input('请描述您的爱车车况（车况优秀, 车况良好, 车况一般, 车况较差, 车况极差）：') or '车况良好'
    sell_times = float(input('请输入过户次数：') or 0)
    car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                     'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

    return car_info_from_customer


def get_car_case(sql_to_customer_carConfig, sql_to_brand, sql_to_system, sql_to_CarConfig_CarCase):
    '''
    获取车辆案例信息
    :param sql_to_customer_carConfig: 查询用户车辆配置信息的SQL
    :param sql_to_CarConfig_CarCase: 查询车辆案例信息及车辆配置的SQL   
    :return: 查询案例结果
    '''
    # 获取客户输入车辆信息
    car_info_from_customer = get_car_info_from_customer()

    if car_info_from_customer['meter_mile'] > 55:
        print('\n客观，您的爱车接近报废啦。。。。。')
    else:
        # 查询客户车辆参数配置、类型
        select = SelectMySQL(host='192.168.0.3',
                             user='clean',
                             passwd='Zlpg1234!',
                             db='valuation_web')
        # 根据车辆品牌、车系、车型获取用户车辆参数配置信息
        customer_carConfig = select.get_df(sql_to_customer_carConfig.format(car_info_from_customer['car_brand'],
                                                                            car_info_from_customer['car_system'],
                                                                            car_info_from_customer['car_model']))
        # 查询车辆类型
        car_class = customer_carConfig[0]['car_class']

        # 查询所有品牌、车系
        car_brand = select.get_df(sql_to_brand.format(car_class))
        car_system = select.get_df(sql_to_system.format(car_class))
        brands = [i['car_brand'] for i in car_brand]
        systems = [i['car_system'] for i in car_system]

        # 获取案例信息
        car_case = select.get_df(sql_to_CarConfig_CarCase.format(car_class, model_year_dict[car_class]))
        # 将同类车辆案例信息写入DataFrame
        car_case_df = pd.DataFrame(car_case, columns=car_case[0].keys())

        return car_class, car_case_df, brands, systems


def write2csv(df, batch_size, car_class):
    '''
    将数据框分批次写入多个csvs
    :param batch_size: 每批次写入样本数量
    '''
    # train_df, test_df = split_train_(df, 0.2)
    epoch = math.ceil(df.shape[0] / batch_size)
    print('\n**********************开始写入CSV文件*****************************')
    # test_df.to_csv('/home/kdd/python/car/test/%s_test.csv'%car_class, encoding='utf-8', chunksize=1000, index=False)
    for i in range(epoch):
        data = df[i * batch_size: (i + 1) * batch_size]
        data.to_csv(f'/home/kdd/python/car/{car_class}/{car_class}_{i}.csv',
                    encoding='utf-8', chunksize=10000, index=False)  # 写入csv
    print('\n**********************CSV文件写入完成*****************************')



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
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{0}'
	AND 
	    n.model_year >= '{1}'
"""

sql_to_brand = """
select
	distinct car_brand
from
	new_car_information
where
	car_class = '{0}'
order by id
"""

sql_to_system = """
select
	distinct car_system
from
	new_car_information
where
	car_class = '{0}'
order by id
"""


# 非纯电动特征名称
col_NEV = ['car_brand', 'car_system', 'cylinder_number', 'driving', 'gearbox_type', 'intake_form',
           'maximum_power', 'register_time', 'meter_mile', 'sell_times', 'vendor_guide_price',
           'model_year', 'price']
# 纯电动特征名称
col_EV = ['car_brand', 'car_system', 'driving', 'gearbox_type','maximum_power', 'voyage_range', 'register_time',
          'meter_mile', 'sell_times', 'vendor_guide_price', 'model_year', 'price']


# 用于过滤太老旧的车
model_year_dict = {'saloon': 2005, 'suv': 2007, 'mpv': 2006, 'minibus': 2007, 'supercar': 0, 'EV': 0}



if __name__ == '__main__':

    # 1、根据用户输入的车辆信息获取同一类型的车辆案例信息
    car_class, car_case_df, brands, systems = get_car_case(sql_to_customer_carConfig, sql_to_brand, sql_to_system,
                                                      sql_to_CarConfig_CarCase)

    # 2、对案例信息进行处理
    process = Processing()
    categories, col_categ = process.get_category(car_class, brands, systems)
    df_preprocessed = process.preprocess(car_case_df)  # 预处理
    df_disrete = process.feature_encode(df_preprocessed, col_NEV, col_EV) # 离散化
    df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # onehot编码
    df = pd.concat([df_categ, df_preprocessed[['meter_mile', 'vendor_guide_price', 'price']]], axis=1)
    print(df.shape)

    # 3、将处理后的数据写入CSV文件
    write2csv(df=df, batch_size=50000, car_class=car_class)



