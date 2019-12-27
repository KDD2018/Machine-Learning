#!/usr/bin/python3
# -*- coding: utf-8 -*-


from datetime import datetime
from selectMySQL import SelectMySQL
import pandas as pd
import tensorflow as tf
from processing import Processing
import numpy as np



def get_car_info_from_customer():
    '''
    获取用户提供的车辆信息
    :return: 用户车辆表
    '''

    brand_name = input('请输入品牌：') or '大众'
    car_system = input('请输入车系：') or '途观'
    car_model_name = input('请输入车型：') or '2016款 280TSI 自动两驱丝绸之路舒适版'
    register_time = datetime.strptime(input('请输入上牌时间：') or '2016-09-01', '%Y-%m-%d')
    meter_mile = float(input('请输入已行驶里程（公里）：') or 300000)
    sell_times = float(input('请输入过户次数：') or 0)
    car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                     'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

    return car_info_from_customer


def get_customer_car_df(sql_to_customer_carConfig, sql_to_brand, sql_to_system):
    '''
    获取用户车辆的全部信息
    :param sql_to_customer_carConfig: 查询用户车辆参数配置的SQL
    :return: 用户车辆全部信息表
    '''
    # 获取客户输入车辆信息
    car_info_from_customer = get_car_info_from_customer()

    if car_info_from_customer['meter_mile'] > 550000:
        print('\n客观，您的爱车接近报废啦。。。。。')
    else:
        # 查询客户车辆参数配置、类型
        select = SelectMySQL(host='192.168.0.3', user='clean', passwd='Zlpg1234!', db='valuation_web')
        # 根据车辆品牌、车系、车型获取用户车辆参数配置信息
        customer_carConfig = select.get_df(sql_to_customer_carConfig.format(car_info_from_customer['car_brand'],
                                                                            car_info_from_customer['car_system'],
                                                                            car_info_from_customer['car_model']))
        if customer_carConfig:
            # 合并客户车辆配置信息和用户输入信息
            customer_car_info = dict(car_info_from_customer, **customer_carConfig[0])
            customer_car_info['price'] = 1  # 默认给出客户汽车价格为1元，方便与案例共用一套特征处理的方法
            # 客户汽车所属类别
            car_class = customer_car_info['car_class']
            print(f'\n客官，您的爱车类型属于：{car_class}\n')

            if car_class not in ['saloon', 'suv', 'mpv', 'supercar', 'minibus', 'EV']:
                print('\n客官，现不支持此类车型的估值')
            else:
                # 查询所有品牌、车系
                car_brand = select.get_df(sql_to_brand.format(car_class))
                car_system = select.get_df(sql_to_system.format(car_class))
                brands = [i['car_brand'] for i in car_brand]
                systems = [i['car_system'] for i in car_system]

                # 将客户车辆信息写入DataFrame
                customer_car_df = pd.DataFrame([customer_car_info])

                return customer_car_df, car_class, brands, systems
        else:
            print('\n客官，查无此车型')
            return None, None, None, None


def load_model(car_class):
    '''
    加载模型
    :param car_class: 车辆所属类型
    :return: 权重、偏置
    '''
    w = b = None
    # 模型路径
    model_dir = f'../../model-param/{car_class}'
    with tf.Session() as sess:
        # 加载模型文件名
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 导入最新模型图结构
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)

            graph = tf.get_default_graph()

            # 家在权重和偏置
            w = sess.run(graph.get_tensor_by_name('Model/weight:0'))
            b = sess.run(graph.get_tensor_by_name('Model/bias:0'))
        else:
            print('\n客官，尚未发现此类车型的模型文件。。。')
        return w, b


def predict(X, weight, bias):
    '''
    预测
    :param X: 待测样本特征
    :param weight: 学得权重
    :param bias: 学得偏置
    :return: 预测值
    '''
    cp = np.dot(X, weight) + bias
    price = np.expm1(cp)

    return price


def run():

    # 1、获取用户车辆的全部信息
    customer_car_df, car_class, brands, systems = get_customer_car_df(sql_to_customer_carConfig, sql_to_brand, sql_to_system)

    if car_class:
        # 2、根据车辆类型加载模型模型
        weight, bias = load_model(car_class)

        # 3、对用户车信息进行处理
        process = Processing()
        categories, col_categ = process.get_category(car_class, brands, systems)
        df_preprocessed = process.preprocess(customer_car_df)  # 预处理
        df_disrete = process.feature_encode(df_preprocessed, col_NEV, col_EV)  # 离散化
        df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # onehot编码
        df = pd.concat([df_categ, df_preprocessed[['meter_mile', 'vendor_guide_price']]], axis=1)
        df = df.astype('float32')

        # 4、预测用户车辆价值
        customer_car_price = predict(df, weight, bias)
        print(f'\n客官，您的爱车值这个价：{customer_car_price}')

def score(y, y_hat):
    sse = sum((y - y_hat)**2)
    sst = sum((y - np.average(y_hat, axis=0))**2)
    r = 1- (sse / sst)
    return r

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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':

    # run()
    car_class = 'suv'

    # 获取测试集数据
    test_data = pd.read_csv(f'/home/kdd/python/car/{car_class}_test/{car_class}.csv', sep=',', encoding='utf-8',
                            skiprows=0)
    print(test_data.shape)
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    print(y_test.shape)

    #加载模型
    weight, bias = load_model(car_class)

    # 预测
    y_hat = predict(x_test, weight, bias)
    y_hat = y_hat.flatten()
    print(y_hat.shape)

    score = score(y_test, y_hat)
    print(f'{car_class}拟合优度为：{score}')