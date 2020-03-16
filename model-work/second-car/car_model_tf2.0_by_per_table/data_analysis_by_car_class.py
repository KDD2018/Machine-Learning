#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymysql
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


class Analysis(object):
    '''分析各类型中数据分桶'''
    def __init__(self):
        self.default_conn = pymysql.connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!',
                                            db='valuation_web', port=3306, charset='utf8',
                                            cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.default_cursor = self.default_conn.cursor()


    def analysis(self):
        car_class = 'SUV'
        '''
              sql_to_case_by_class = """SELECT s.car_brand, s.car_system, s.displacement,s.driving, s.maximum_power, 
                                      s.voyage_range, s.level, s.normalized_vendor_guide_price, s.model_year, 
                                      s.register_time, s.meter_mile, s.sell_times, s.price FROM 
                                      second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                                      s.car_model_id = n.car_model_id WHERE s.isdeleted = 0"""
        '''
        sql_to_case_by_class = """SELECT register_date,meter_mile, mile_per_year, price from second_car_sell_{0}"""
        self.default_cursor.execute(sql_to_case_by_class.format(car_class))
        case = self.default_cursor.fetchall()
        self.default_cursor.close()
        self.default_conn.close()
        df =  pd.DataFrame(case)
        #print(df.shape)
        #print(df.shape[1:])
        plt.figure()
        plt.scatter(df['mile_per_year'], df['price'])
        plt.xlabel('mile_per_year')
        plt.ylabel('price')
        plt.title(f'car_class={car_class}')
        plt.show()


    def preprocess(self, df, car_class):
        '''
        预处理
        :param df: 
        :return: 
        '''
        df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'] / 10000.0  # 换算成万元
        df.loc[:, 'meter_mile'] = 1 - df['meter_mile'] / 60  # 转换成行驶里程成新率
        df.loc[:, 'register_time'] = df['register_time'].map(
            lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
        df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'].map(lambda x: np.log1p(x))
        if car_class == 'EV':
            df.loc[:, 'voyage_range'] = df['voyage_range'] / 1000.0  # 换算成公里
        else:
            df.pop('voyage_range')
        del df['car_class'], df['car_model']

        return df


    def plt_plot(self):
        df = self.analysis()
        plt.figure()
        plt.scatter(df['meter_mile'], df['price'])
        plt.show()



if __name__ == '__main__':
    ana = Analysis()
    ana.analysis()