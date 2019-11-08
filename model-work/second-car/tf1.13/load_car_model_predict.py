#!/usr/bin/python3
# -*- coding: utf-8 -*-


from datetime import datetime
from selectMySQL import SelectMySQL
import pandas as pd
import tensorflow as tf



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


def customer_car_df(sql_to_customer_carConfig):
    '''
    获取用户车辆的全部信息
    :param sql_to_customer_carConfig: 查询用户车辆参数配置的SQL
    :return: 用户车辆全部信息表
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
        if customer_carConfig:
            # 合并客户车辆配置信息和用户输入信息
            customer_car_info = dict(car_info_from_customer, **customer_carConfig[0])
            # 客户汽车所属类别
            car_class = customer_car_info['car_class']
            print(f'\n客官，您的爱车类型属于：{car_class}\n')

            if car_class not in ['saloon', 'suv', 'mpv', 'supercar', 'minibus', 'EV']:
                print('\n客官，现不支持此类车型的估值')
            else:
                # 将客户车辆信息写入DataFrame
                customer_car_df = pd.DataFrame([customer_car_info])

                return customer_car_df, car_class
        else:
            print('\n客官，查无此车型')


def load_model(car_class):
    '''
    加载模型
    :param car_class: 车辆所属类型
    :return: 权重、偏置
    '''
    model_dir = f'../model-param/{car_class}/{car_class}.ckpt'
    with tf.Session() as sess:
        # 加载模型文件名
        ckpt = tf.train.get_checkpoint_state(model_dir)

        # 导入最新模型图结构
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()

        # 家在权重和偏置
        w = sess.run(graph.get_tensor_by_name('Model/weight:0'))
        b = sess.run(graph.get_tensor_by_name('Model/bias:0'))

    return w, b


def predict(X, weight, bias):
    '''
    预测
    :param X: 待测样本特征
    :param weight: 学得权重
    :param bias: 学得偏置
    :return: 预测值
    '''
    return tf.matmul(X, weight) + bias


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
pd.set_option('display.max_columns', None)



if __name__ == '__main__':

    # 1、获取用户车辆的全部信息
    customer_car_df, car_class = customer_car_df(sql_to_customer_carConfig)
    print(customer_car_df)

    # 2、根据车辆类型加载模型模型
    weight, bias = load_model(car_class)

    # 3、对用户车信息进行处理
    # processing customer_car_df  此处调用Processing类

    # 4、预测用户车辆价值
    customer_car_price = predict(customer_car_df, weight, bias)

