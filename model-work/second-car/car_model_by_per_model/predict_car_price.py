#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import datetime
import pymysql
from processing import Processing
import joblib


class Predict(object):
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
        brand_name = input('\n请输入品牌：') or '北汽威旺'
        car_system = input('请输入车系：') or '北汽威旺M20'
        car_model_name = input('请输入车型：') or '2014款 1.5L实用型BJ415A'
        register_time = datetime.datetime.strptime(input('请输入上牌时间：') or '2018-03-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（万公里）：') or 1.31)
        sell_times = float(input('请输入过户次数：') or 0)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                                  'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

        return car_info_from_customer


    def get_customer_car_all_info(self, car_info_from_customer):
        '''根据用户车辆型号获取车辆配置信息'''

        # 获取用户车辆型号及使用情况
        #car_info_from_customer = self.get_car_info_from_customer()
        # print(car_info_from_customer)
        # 查询用户车辆的配置信息
        sql_to_customer_car_config = """SELECT car_system_id, car_model_id, displacement, driving, maximum_power, 
                                     voyage_range, model_year, car_class, vendor_guide_price from new_car_information_t 
                                     where car_brand = '{0}' and car_system = '{1}' and car_model = '{2}'"""
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


    def predict_by_car_model(self, customer_car_info):
        """
        加载车型模型预测或者根据车型案例加以修正估值
        :param customer_car_info: 用户车辆信息字典
        :return: 用户车辆价格
        """
        # 查询车型对应的案例信息
        sql_to_sample_size_by_car_model = """SELECT COUNT(*) sample_size FROM second_car_sell_{0} s INNER JOIN 
                                             new_car_information_t n ON s.car_model_id = n.car_model_id WHERE 
                                             n.car_model_id = '{1}' 
                                             and s.isdeleted = 0 """
        self.cursor.execute(sql_to_sample_size_by_car_model.format(customer_car_info['car_class'],
                                                                   customer_car_info['car_model_id']))
        sample_size_current_car_model = self.cursor.fetchall()[0]['sample_size']
        if sample_size_current_car_model >= 200:
            try:
                regressor = joblib.load(
                    f"./model_params/{customer_car_info['car_brand']}-{customer_car_info['car_brand']}-{customer_car_info['car_model']}.joblib")
                # 预处理用户车辆的特征信息
                customer_car_df = self.preprocess_customer_car_info(customer_car_info)
                # print(customer_car_df)
                # 获取用户车辆的厂商指导价
                vendor_guide_price = customer_car_df.pop('vendor_guide_price')[0]
                car_models = [customer_car_df['car_model']]
                customer_car_df = Processing().feature_engineering(customer_car_info['car_class'], customer_car_df, car_models)
                customer_car_df.pop('hedge_ratio')
                hedge_ratio = regressor.predict(customer_car_df)
                customer_car_info['predicted_price'] = hedge_ratio * vendor_guide_price / 10000
                customer_car_info['label'] = '车型模型'
                #print(f'\n您的爱车值这个价: {customer_car_price} 万元')
            except:
                customer_car_info = self.predict_by_market_approach(customer_car_info)
        else:
            customer_car_info = self.predict_by_market_approach(customer_car_info)
        return customer_car_info


    def predict_by_car_system(self, customer_car_info):
        """
        根据车系模型预测
        :param customer_car_info: 用户车辆信息字典
        :return: 用户车辆价格
        """
        # 查询对应车系的样本容量
        sql_to_sample_size_by_car_system = """SELECT count(*) sample_size FROM second_car_sell_{0} s INNER JOIN 
                                           new_car_information_t n ON s.car_model_id = n.car_model_id WHERE 
                                           n.car_system_id = '{1}' and s.isdeleted = 0"""
        self.cursor.execute(sql_to_sample_size_by_car_system.format(customer_car_info['car_class'],
                                                                    customer_car_info['car_system_id']))
        sample_size_current_car_system = self.cursor.fetchall()[0]['sample_size']
        if sample_size_current_car_system >= 500:
            try:
                regressor = joblib.load(
                    f"./model_params/{customer_car_info['car_brand']}-{customer_car_info['car_system']}.joblib")
                # 查询车系下车型集合
                sql_to_car_models = """SELECT DISTINCT n.car_model FROM new_car_information_t n INNER JOIN 
                                    second_car_sell_{0} s ON s.car_model_id = n.car_model_id WHERE 
                                    n.car_system_id = '{1}' and s.isdeleted = 0 ORDER BY n.id"""
                self.cursor.execute(sql_to_car_models.format(customer_car_info['car_class'],
                                                             customer_car_info['car_system_id']))
                car_models = self.cursor.fetchall()
                car_models = [car_model_dict['car_model'] for car_model_dict in car_models]
                # 预处理用户车辆的特征信息
                customer_car_df = self.preprocess_customer_car_info(customer_car_info)
                # print(customer_car_df)
                # 获取用户车辆的厂商指导价
                vendor_guide_price = customer_car_df.pop('vendor_guide_price')[0]
                customer_car_df = Processing().feature_engineering(customer_car_info['car_class'],
                                                                   customer_car_df,
                                                                   car_models)
                customer_car_df.pop('hedge_ratio')
                # print(customer_car_df.head(1))
                hedge_ratio = regressor.predict(customer_car_df)[0]
                customer_car_info['predicted_price'] = hedge_ratio * vendor_guide_price / 10000
                customer_car_info['label'] = '车系模型'
                #print(f'\n您的爱车值这个价: {customer_car_price} 万元')
            except:
                customer_car_info = self.predict_by_car_model(customer_car_info)
        else:
            customer_car_info = self.predict_by_car_model(customer_car_info)
        return customer_car_info


    def predict_by_market_approach(self, customer_car_info):
        '''
        根据市场法对案例加以修正估值
        :param customer_car_info: 用户车辆信息字典
        :return: 用户车辆价格
        '''
        print(customer_car_info['car_class'])
        # 计算用户车辆的车龄
        customer_car_info['car_age'] = (datetime.datetime.today() - customer_car_info['register_time']).days / 365
        # 计算用户车辆的行驶里程成新率
        customer_car_info['mileage_newness_rate'] = 1 - customer_car_info['meter_mile'] / 60
        # 查询对应车型的案例信息
        sql_to_samples_by_car_model = """SELECT s.id, s.car_age, s.mile_per_year, s.mileage_newness_rate, s.sell_times, 
                                      s.price FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                      s.car_model_id = n.car_model_id WHERE n.car_model_id = '{1}' AND s.isdeleted = 0"""
        self.cursor.execute(sql_to_samples_by_car_model.format(customer_car_info['car_class'],
                                                               customer_car_info['car_model_id']))
        samples = self.cursor.fetchall()
        # print(len(samples))
        adjusted_prices = []
        for sample in samples:
            if np.abs(customer_car_info['car_age'] - sample['car_age']) > 0.5:
                continue

            # 计算行驶里程成新率的相对变化率
            meter_mile_rate = (customer_car_info['mileage_newness_rate'] - sample['mileage_newness_rate']) / sample['mileage_newness_rate']
            adjusted_price = sample['price'] * (1 + meter_mile_rate)

            adjusted_prices.append(adjusted_price)
        if adjusted_prices:
            # print(adjusted_prices)
            customer_car_info['predicted_price'] = np.median(adjusted_prices)
            customer_car_info['label'] = '案例修正'
        else:
            customer_car_info['predicted_price'] = 0
        return customer_car_info



    def run(self):
        """主程序"""
        try:
            button = int(input('\n客官, 您是要单样本预测还是多样本批量预测(单样本:1,多样本:0): ') or 1)
            sql_to_count_samples = """SELECT COUNT(*) sample_size FROM second_car_sell_{0} WHERE car_model_id = {1} AND 
                                   isdeleted = 0"""
            if button == 1:
                # 单样本预测
                # 获取用户车辆型号及使用情况
                car_info_from_customer = self.get_car_info_from_customer()
                customer_car_info = self.get_customer_car_all_info(car_info_from_customer)
                print(customer_car_info)
                self.cursor.execute(sql_to_count_samples.format(customer_car_info['car_class'],
                                                                customer_car_info['car_model_id']))
                current_sample_size = self.cursor.fetchall()[0]['sample_size']
                if current_sample_size > 0:
                    customer_car_info = self.predict_by_car_system(customer_car_info)
                    print(f"\n您的爱车值这个价: {customer_car_info['predicted_price']} 万元")
                else:
                    print('\n暂无该车型案例.')

            elif button == 0:
                # 多样本批量预测
                sql_to_test_samples = """select n.car_brand, n.car_system, n.car_model, s.register_time, 
                                         (s.meter_mile / 10000) as meter_mile, s.sell_times, (s.price / 10000) as price 
                                         from second_car_sell_guazi s inner join new_car_information_t n 
                                         on s.car_model_id = n.car_model_id where s.car_class != 'truck'"""
                self.cursor.execute(sql_to_test_samples)
                test_case = self.cursor.fetchall()

                test_case_list = []
                for car_info_from_customer in test_case:
                    customer_car_info = self.get_customer_car_all_info(car_info_from_customer)
                    if customer_car_info['car_class'] in ['suv', 'saloon', 'minibus', 'mpv', 'ev', 'supercar']:
                        self.cursor.execute(sql_to_count_samples.format(customer_car_info['car_class'],
                                                                        customer_car_info['car_model_id']))
                        current_sample_size = self.cursor.fetchall()[0]['sample_size']
                        if current_sample_size > 0:
                            # 预测用户车辆价格
                            customer_car_info = self.predict_by_car_system(customer_car_info)
                        else:
                            customer_car_info['predict_price'] = 0
                            print('\n暂无该车型案例.')
                    else:
                        customer_car_info['predict_price'] = 0

                    test_case_list.append(customer_car_info)
                pd.DataFrame(test_case_list).to_csv('./second_car_test.csv', encoding='utf-8')
            else:
                print('\n请您输入正确信息!!!')
        except ValueError as ve:
            print('\n请输入0或者1......')

        self.conn.commit()
        self.cursor.close()
        self.conn.close()



pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    predict = Predict()
    predict.run()