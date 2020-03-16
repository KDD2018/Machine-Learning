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
from sklearn import linear_model, tree
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


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
        brand_name = input('\n请输入品牌：') or '江淮'
        car_system = input('请输入车系：') or '瑞风S7'
        car_model_name = input('请输入车型：') or '2016款 1.5T CVT精英版'
        register_time = datetime.datetime.strptime(input('请输入上牌时间：') or '2018-08-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（万公里）：') or 2.13)
        sell_times = float(input('请输入过户次数：') or 0)
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

    def preprocess_customer_car_info(self, car_info):
        '''
        对用户车辆进行预处理
        :param car_info: 用户车辆型号及使用情况
        :return: 预处理后的用户车辆信息
        '''
        car_info_df = pd.DataFrame([car_info])
        car_info_df.loc[:, 'car_age'] = car_info_df.loc[:, 'register_time'].map(
            lambda x: (datetime.datetime.today() - x).days / 365)
        car_info_df.loc[:, 'mile_per_year'] = car_info_df.loc[:, 'meter_mile'] / car_info_df.loc[:, 'car_age']
        car_info_df.loc[:, 'mileage_newness_rate'] = 1 - car_info_df.loc[:, 'meter_mile'] / 60
        car_info_df.loc[:, 'hedge_ratio'] = 0
        for col in ['car_brand', 'car_system', 'car_class', 'register_time', 'meter_mile']:
            car_info_df.pop(col)

        return car_info_df


    def get_case_by_car_system(self, car_class, customer_car_info):
        '''
        获取当前车系下的车型集合和样本案例
        :param car_class: 车辆类型
        :param customer_car_info: 用户车辆信息字典
        :return: 车系下车型集和样本案例
        '''
        # 查询车系下车型集合
        sql_to_car_models = """SELECT DISTINCT n.car_model FROM new_car_information_t n INNER JOIN second_car_sell_{0} s 
                               ON s.car_model_id = n.car_model_id WHERE n.car_brand = '{1}' and n.car_system = '{2}' and 
                               s.isdeleted = 0 ORDER BY n.id"""
        self.cursor.execute(sql_to_car_models.format(car_class,
                                                     customer_car_info['car_brand'],
                                                     customer_car_info['car_system']))
        car_models = self.cursor.fetchall()
        car_models = [car_model_dict['car_model'] for car_model_dict in car_models]
        # 查询对应车系的案例信息
        sql_to_samples_by_car_system = """SELECT s.car_model, s.displacement, s.driving, s.maximum_power, s.voyage_range, 
                                          s.model_year, s.car_age, s.mile_per_year, s.mileage_newness_rate, s.sell_times, 
                                          s.hedge_ratio FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                          s.car_model_id = n.car_model_id WHERE n.car_brand = '{1}' and 
                                          n.car_system = '{2}' and s.isdeleted = 0 """
        self.cursor.execute(sql_to_samples_by_car_system.format(car_class,
                                                                customer_car_info['car_brand'],
                                                                customer_car_info['car_system']))
        samples = self.cursor.fetchall()

        return car_models, samples


    def get_case_by_car_model(self, car_class, customer_car_info):
        '''
        获取当前车型下的样本案例
        :param car_class: 车辆类型
        :param customer_car_info: 用户车辆信息字典
        :return: 车型样本案例
        '''
        # 查询对应车型的案例信息
        sql_to_samples_by_car_model = """SELECT s.car_model, s.displacement, s.driving, s.maximum_power, s.voyage_range, 
                                         s.model_year, s.car_age, s.mile_per_year, s.mileage_newness_rate, s.sell_times, 
                                         s.hedge_ratio FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                         s.car_model_id = n.car_model_id WHERE n.car_brand = '{1}' AND 
                                         n.car_system = '{2}' AND n.car_model = '{3}' AND s.isdeleted = 0 """
        self.cursor.execute(sql_to_samples_by_car_model.format(car_class,
                                                               customer_car_info['car_brand'],
                                                               customer_car_info['car_system'],
                                                               customer_car_info['car_model']))
        samples = self.cursor.fetchall()

        return samples


    def feature_engineering(self, car_class, data, car_models):
        '''
        特征工程
        :param car_class: 车辆类型
        :param data: 案例信息
        :param car_models: 车型集合
        :return: 
        '''
        if car_class == 'EV':
            data.pop('displacement')
        else:
            data.pop('voyage_range')

        data.dropna(inplace=True)
        data.index = range(data.shape[0])
        #print(data.shape)

        # 处理数据
        process = Processing()
        # 获取分类型特征及特征类别
        categories, categorical_features = process.get_category(car_class, car_models)
        #print(categorical_features)
        # 分类型特征离散化
        df_disrete = process.feature_encode(data, car_class)
        #print(df_disrete.isnull().any())
        # print(df_disrete[df_disrete['voyage_range'].isnull()])
        #print(df_disrete.head(1)) ['car_model', 'displacement', 'driving', 'maximum_power', 'sell_times', 'model_year']
        # one-hot编码
        df_categ = process.onehot_encode(df_disrete[categorical_features], categories)
        df = pd.concat([df_categ, df_disrete[['car_age', 'mile_per_year',
                                              'mileage_newness_rate', 'hedge_ratio']]], axis=1)

        return df


    def build_and_save_model(self, df, car_brand, car_system, *args):
        '''
        训练保存模型
        :param df: 训练数据
        :param test_df: 待预测的用户车辆信息
        :param vendor_guide_price: 用户车辆的厂商指导价
        :return: 用户车辆价格
        '''
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

        if score >= 0.9:
            if args:
                joblib.dump(regressor, f'./model_params/{car_brand}-{car_system}-{args[0]}.joblib')
            else:
                joblib.dump(regressor, f'./model_params/{car_brand}-{car_system}.joblib')
        else:
            print('\n当前模型拟合优度较低...')
        # 预测用户车辆当前的保值率
        #hedge_ratio = regressor.predict(test_df)[0]
        #print(f'您的爱车当前保值率为: {hedge_ratio}')
        # 计算用户车辆当前价格
        #predicted_price = hedge_ratio * vendor_guide_price /10000
        #print(f'\n您的车值这个价: {predicted_price:.2f} 万元')


    def run(self):
        '''主程序'''
        customer_car_info = self.get_customer_car_all_info()
        # 车辆类型
        car_class = customer_car_info['car_class']
        # 预处理用户车辆的特征信息
        #customer_car_df = self.preprocess_customer_car_info(customer_car_info)
        # 获取用户车辆的厂商指导价
        #vendor_guide_price = customer_car_df.pop('vendor_guide_price')[0]
        # 查询对应车系的样本容量
        sql_to_sample_size_by_car_system = """SELECT count(*) sample_size FROM second_car_sell_{0} s INNER JOIN 
                                              new_car_information_t n ON s.car_model_id = n.car_model_id WHERE 
                                              n.car_brand = '{1}' and n.car_system = '{2}' and s.isdeleted = 0 """
        self.cursor.execute(sql_to_sample_size_by_car_system.format(car_class,
                                                                    customer_car_info['car_brand'],
                                                                    customer_car_info['car_system']))
        sample_size_current_car_system = self.cursor.fetchall()[0]['sample_size']
        print(sample_size_current_car_system)
        if sample_size_current_car_system >= 500:
            # 查询对应车系的案例信息
            car_models, samples = self.get_case_by_car_system(car_class, customer_car_info)
            sample_df = pd.DataFrame(samples)
            # 特征工程
            df = self.feature_engineering(car_class, sample_df, car_models)
            # 训练并保存模型
            self.build_and_save_model(df, customer_car_info['car_brand'], customer_car_info['car_system'])

        else:
            sql_to_sample_size_by_car_model = """SELECT COUNT(*) sample_size FROM second_car_sell_{0} s INNER JOIN 
                                                 new_car_information_t n ON s.car_model_id = n.car_model_id WHERE 
                                                 n.car_brand = '{1}' and n.car_system = '{2}' and n.car_model = '{3}' 
                                                 and s.isdeleted = 0 """
            self.cursor.execute(sql_to_sample_size_by_car_model.format(car_class,
                                                                       customer_car_info['car_brand'],
                                                                       customer_car_info['car_system'],
                                                                       customer_car_info['car_model']))
            sample_size_current_car_model = self.cursor.fetchall()[0]['sample_size']
            print(sample_size_current_car_model)
            if sample_size_current_car_model >= 120:
                # 查询对应车型的样本案例
                samples = self.get_case_by_car_model(car_class, customer_car_info)
                sample_df = pd.DataFrame(samples)
                car_models = [customer_car_info['car_model']]
                # 特征工程
                df = self.feature_engineering(car_class, sample_df, car_models)
                # 训练并保存模型
                self.build_and_save_model(df, customer_car_info['car_brand'],
                                          customer_car_info['car_system'], customer_car_info['car_model'])
            else:
                # 根据车型案例进行加以修正
                print('\n样本案例缺少,无法运用机器学习算法,采用案例修正法.')

        # print(df.head(1))
        #df = df.append(customer_car_df, ignore_index=True, sort=False)
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        '''
        df, car_class, car_models, vendor_guide_price = self.get_case_df()
        df = self.feature_engineering(car_class, df, car_models)
        #print(df.shape)
        customer_car = df.iloc[-1:]
        customer_car.pop('hedge_ratio')
        #print(customer_car.shape)
        df = df.iloc[:-1]
        self.build_model(df, customer_car, vendor_guide_price)
        '''




pd.set_option('display.max_columns', None)


if __name__ == '__main__':

    model = Model()
    model.run()

