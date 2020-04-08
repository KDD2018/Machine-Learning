#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import pandas as pd
#from datetime import datetime
import datetime
import pymysql
import matplotlib.pyplot as plt
from processing import Processing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import joblib



class Model(object):
    '''根据车系和车型建立车辆估值模型'''
    def __init__(self):
        '''初始化MySQL连接'''
        self.conn = pymysql.connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!',
                                           db='valuation_web', port=3306, charset='utf8',
                                           cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.cursor = self.conn.cursor()


    def get_car_info_from_customer(self):
        '''
        获取用户提供的车辆信息
        :return: 用户车辆信息字典
        '''
        print('\n客官,请提供您爱车的相关信息：')
        brand_name = input('\n请输入品牌：') or 'MINI'
        car_system = input('请输入车系：') or 'MINI'
        car_model_name = input('请输入车型：') or '2015款 2.0T COOPER S 五门版'
        register_time = datetime.datetime.strptime(input('请输入上牌时间：') or '2015-07-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（万公里）：') or 3.81)
        sell_times = float(input('请输入过户次数：') or 2)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                                  'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

        return car_info_from_customer


    def get_customer_car_all_info(self):
        '''根据用户车辆型号获取车辆配置信息'''
        # 获取用户车辆型号及使用情况
        car_info_from_customer = self.get_car_info_from_customer()
        # print(car_info_from_customer)
        # 查询用户车辆的配置信息
        sql_to_customer_car_config = """SELECT displacement, driving, maximum_power, voyage_range, model_year, car_class, 
                                        vendor_guide_price from new_car_information_t where car_brand = '{0}' and 
                                        car_system = '{1}' and car_model = '{2}'"""
        self.cursor.execute(sql_to_customer_car_config.format(car_info_from_customer['car_brand'],
                                                              car_info_from_customer['car_system'],
                                                              car_info_from_customer['car_model']))
        config = self.cursor.fetchall()
        car_info = dict(car_info_from_customer, **config[0])

        return car_info


    def get_case_by_car_system(self, car_class, sample_size_per_car_system):
        """
        获取当前车系下的车型集合和样本案例
        :param car_class: 车辆类型
        :param car_systems_count: 用户车辆信息字典
        :return: 车系下车型集和样本案例
        """

        # 查询车系下车型集合
        sql_to_car_models = """SELECT DISTINCT n.car_model FROM new_car_information_t n INNER JOIN second_car_sell_{0} s 
                            ON s.car_model_id = n.car_model_id WHERE n.car_system_id = {1} and s.isdeleted = 0 
                            ORDER BY n.id"""
        self.cursor.execute(sql_to_car_models.format(car_class,
                                                     sample_size_per_car_system['car_system_id']))
        car_models = self.cursor.fetchall()
        car_models = [car_model_dict['car_model'] for car_model_dict in car_models]
        # 查询对应车系的案例信息
        sql_to_samples_by_car_system = """SELECT s.car_model, s.displacement, s.driving, s.maximum_power, s.voyage_range, 
                                       s.model_year, s.car_age, s.mile_per_year, s.mileage_newness_rate, s.sell_times, 
                                       s.hedge_ratio FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                       s.car_model_id = n.car_model_id WHERE n.car_system_id = {1} and s.isdeleted = 0"""
        self.cursor.execute(sql_to_samples_by_car_system.format(car_class,
                                                                sample_size_per_car_system['car_system_id']))
        samples = self.cursor.fetchall()
        samples_df = pd.DataFrame(samples)

        return car_models, samples_df


    def get_case_by_car_model(self, car_class, sample_size_per_car_model):
        '''
        获取当前车型下的样本案例
        :param car_class: 车辆类型
        :param customer_car_info: 用户车辆信息字典
        :return: 车型样本案例
        '''
        # 查询对应车型的案例信息
        sql_to_samples_by_car_model = """SELECT s.car_age, s.mile_per_year, s.mileage_newness_rate, s.sell_times, 
                                      s.hedge_ratio FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                      s.car_model_id = n.car_model_id WHERE n.car_model_id = {1} AND s.isdeleted = 0"""
        self.cursor.execute(sql_to_samples_by_car_model.format(car_class,
                                                               sample_size_per_car_model['car_model_id']))
        samples = self.cursor.fetchall()
        samples_df = pd.DataFrame(samples)

        return samples_df


    def build_and_save_model(self, df):

        """
        训练保存模型
        :param df: 训练数据
        :return: 模型
        """
        # 拆分特征标签
        y = df.pop('hedge_ratio')
        X = df
        # 拆分测试集和验证集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        # 建立Ridge模型
        regressor = linear_model.Ridge(alpha=0.1)
        reg = regressor.fit(X_train, y_train)
        # 计算模型拟合优度
        score = reg.score(X_test, y_test)
        print(f'\n模型的拟合优度为: {score:2.2%}')

        return regressor, score


    def generate_model_by_car_model(self, car_class, sample_size_per_car_system, model_count_dict):
        """
        根据车型案例生成模型
        :param car_class: 车型类别
        :param car_systems_count: 车系案例字典
        :return: 
        """
        # 查询车型对应的样本案例数量
        sql_to_sample_size_per_car_model = """SELECT n.car_brand, n.car_system, n.car_model, n.car_model_id, 
                                           COUNT(*) sample_size FROM second_car_sell_{0} s INNER JOIN 
                                           new_car_information_t n ON s.car_model_id = n.car_model_id WHERE 
                                           n.car_system_id = {1} and s.isdeleted = 0 GROUP BY s.car_model"""
        self.cursor.execute(sql_to_sample_size_per_car_model.format(car_class,
                                                                    sample_size_per_car_system['car_system_id']))
        sample_size_per_car_model_dict_list = self.cursor.fetchall()
        # 根据车型案例数量选择估值方式
        for sample_size_per_car_model in sample_size_per_car_model_dict_list:
            if sample_size_per_car_model['sample_size'] >= 200:
                # 查询对应车型的样本案例
                samples_df = self.get_case_by_car_model(car_class, sample_size_per_car_model)
                # sample_df = pd.DataFrame(samples)
                # 特征工程
                samples_df.loc[:, 'sell_times'] = pd.cut(samples_df.sell_times, bins=[-1, 0.01, 1, 2, 3, 20],
                                                        labels=['0次', '1次', '2次', '3次', '4次及以上'])  # 左开右闭
                # samples_df.pop('sell_times')
                df = pd.get_dummies(samples_df)
                # 训练模型
                regressor, score = self.build_and_save_model(df)
                if score >= 0.9:
                    model_count_dict['car_model'] = model_count_dict['car_model'] + 1
                    # 模型持久化
                    joblib.dump(regressor,
                                f"./model_params/{sample_size_per_car_model['car_brand']}-{sample_size_per_car_model['car_system']}-\
                                {sample_size_per_car_model['car_model']}.joblib")
                else:
                    model_count_dict['market_approach'] = model_count_dict['market_approach'] + 1
                    print('\n车系案例不足, 车型模型拟合优度不足90%, 采用案例修正法.')
            else:
                model_count_dict['market_approach'] = model_count_dict['market_approach'] + 1
                # 根据车型案例进行加以修正
                print('\n车系、车型样本案例不足,无法运用机器学习算法, 采用案例修正法.')


    def run(self):
        '''主程序'''
        model_count_dict = {'car_system': 0, 'car_model': 0, 'market_approach': 0}
        for car_class in ['suv', 'saloon', 'minibus', 'mpv', 'ev']:
            # 查询各车系对应样本案例数量
            sql_to_sample_size_per_car_system = """SELECT n.car_brand, n.car_system, n.car_system_id, COUNT(*) 
                                                sample_size FROM second_car_sell_{0} s INNER JOIN new_car_information_t 
                                                n ON s.car_model_id = n.car_model_id WHERE s.isdeleted = 0 GROUP BY 
                                                s.car_system"""
            self.cursor.execute(sql_to_sample_size_per_car_system.format(car_class))
            sample_size_per_car_system_dict_list = self.cursor.fetchall()
            # 根据案例分布情况选择模型生成方式
            for sample_size_per_car_system in sample_size_per_car_system_dict_list:
                if sample_size_per_car_system['sample_size'] >= 500:
                    # 根据车系样本案例训练模型
                    car_models, samples_df = self.get_case_by_car_system(car_class, sample_size_per_car_system)
                    # sample_df = pd.DataFrame(samples)
                    # 特征工程
                    df = Processing().feature_engineering(car_class, samples_df, car_models)
                    # 训练模型
                    regressor, score = self.build_and_save_model(df)
                    if score >= 0.9:
                        # 统计该车系下各车型覆盖多少案例
                        sql_to_car_modle_count = """SELECT count(*) FROM new_car_information_t n INNER JOIN 
                                                 second_car_sell_{0} s ON s.car_model_id = n.car_model_id WHERE 
                                                 car_system_id = {1} and s.isdeleted = 0"""
                        sql_to_count = """SELECT count(*) FROM second_car_sell_{0} s WHERE car_brand = '{1}' and 
                        car_system = '{2}' and isdeleted = 0 GROUP BY car_model_id ORDER BY id"""
                        self.cursor.execute(sql_to_car_modle_count.format(car_class,
                                                                          sample_size_per_car_system['car_system_id']))
                        car_model_count = self.cursor.fetchall()[0]
                        # 车系模型累计覆盖多少车型
                        model_count_dict['car_system'] = model_count_dict['car_system'] + car_model_count['count(*)']
                        # 模型持久化
                        joblib.dump(regressor,
                                    f"./model_params/{sample_size_per_car_system['car_brand']}-{sample_size_per_car_system['car_system']}.joblib")
                    else:
                        # 车系模型拟合优度不足90%,则根据车型样本案例训练模型,若车型模型拟合优度不足90%,则根据车型案例加以修正
                        self.generate_model_by_car_model(car_class, sample_size_per_car_system, model_count_dict)
                else:
                    # 车系样本案例不足, 则根据车型样本案例训练模型
                    self.generate_model_by_car_model(car_class, sample_size_per_car_system, model_count_dict)
        print(model_count_dict)
        self.conn.commit()
        self.cursor.close()
        self.conn.close()


pd.set_option('display.max_columns', None)


if __name__ == '__main__':

    model = Model()
    model.run()

