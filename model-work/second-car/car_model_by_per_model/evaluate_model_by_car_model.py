#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import pandas as pd
from datetime import datetime
import pymysql
import matplotlib.pyplot as plt
from sklearn import linear_model, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class Model(object):
    def __init__(self):
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
        brand_name = input('\n请输入品牌：') or '本田'
        car_system = input('请输入车系：') or '思域'
        car_model_name = input('请输入车型：') or '2016款 220TURBO CVT豪华版'
        register_time = datetime.strptime(input('请输入上牌时间：') or '2019-05-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（万公里）：') or 0.8)
        sell_times = float(input('请输入过户次数：') or 0)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                                  'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

        return car_info_from_customer


    def get_case_df(self):
        car_info_from_customer = self.get_car_info_from_customer()
        # 根据用户车辆信息查询该类车型的案例信息
        sql_to_car_class = """SELECT car_class from new_car_information_t where car_brand = '{0}' and 
                         car_system = '{1}' and car_model = '{2}'"""
        # 查询同一车型的案例
        sql_to_case_by_car_model = """select car_age, meter_mile, mile_per_year, price from 
                         second_car_sell_{0} where car_brand = '{1}' and 
                         car_system = '{2}' and car_model = '{3}'"""

        self.cursor.execute(sql_to_car_class.format(car_info_from_customer['car_brand'],
                                                      car_info_from_customer['car_system'],
                                                      car_info_from_customer['car_model']))
        car_class = self.cursor.fetchall()[0]['car_class']
        self.cursor.execute(sql_to_case_by_car_model.format(car_class,car_info_from_customer['car_brand'],
                                                      car_info_from_customer['car_system'],
                                                      car_info_from_customer['car_model']))
        case = self.cursor.fetchall()
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        df = pd.DataFrame(case)

        return df


    def process(self, data):
        df = data.dropna()
        #print(df.shape)
        df.loc[:, 'price'] = df['price'] / 10000  # 换算万元
        df.loc[:, 'meter_mile'] = 1 - df['meter_mile'] / 600000  # 转换成行驶里程成新率
        df.loc[:, 'register_date'] = df['register_date'].map(
            lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
        df.loc[:, 'price'] = df['price'].map(lambda x: np.log1p(x))
        df.loc[:, 'mile_per_year'] = df.loc[:, 'meter_mile'] / df.loc[:, 'register_time']

        return df

    def run(self):
        df = self.get_case_df()
        #df = self.process(df)
        #print(df.head(5))
        plt.figure()
        plt.scatter(df['mile_per_year'], df['price'])
        plt.xlabel('mile_per_year')
        plt.ylabel('price')
        score = self.fit_model(df)

        plt.show()


    def fit_model(self, df):
        #print(df.isnull().any())
        y = df.pop('price')
        #print(y)
        #scaler = StandardScaler()
        #y = scaler.fit_transform(np.array(y).reshape(len(y), 1))
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        regressor = linear_model.Ridge(alpha=0.1)
        reg = regressor.fit(X_train, y_train)
        score = reg.score(X_test, y_test)
        print(score)


if __name__ == '__main__':

    model = Model()
    model.run()

