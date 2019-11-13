#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


class Processing():
    '''
    预处理、特征工程
    '''
    def preprocess(self, data):
        '''
        数据预处理
        :param data: 要处理的数据框
        :return: 处理后的数据框
        '''
        if data.loc[0, 'car_class'] != 'EV':
            del data['voyage_range']
        data.dropna(inplace=True)
        data.index = range(len(data))
        data.loc[:, 'meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
        data.loc[:, 'price'] = data['price'] / 10000.0  # 换算成万元
        data.loc[:, 'vendor_guide_price'] = data['vendor_guide_price'] / 10000.0  # 换算成万元
        df_ = data.loc[data.meter_mile < 55, :].copy()  # 过滤掉40万公里以上的案例
        df = df_.loc[df_.sell_times <= 10, :].copy()  # 过滤掉过户次数10次以上的
        df.loc[:, 'meter_mile'] = 1 - df['meter_mile'] / 60  # 转换成行驶里程成新率
        df.loc[:, 'register_time'] = df['register_time'].map(
            lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
        df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'].map(lambda x: np.log1p(x))
        df.loc[:, 'price'] = df['price'].map(lambda x: np.log1p(x))

        return df


    def feature_encode(self, data, col1, col2):
        '''
        特征离散编码
        :param data: 数据框
        :return: 离散化处理的数据框
        '''
        if data.loc[0, 'car_class'] != 'EV':
            df = data.loc[:, col1]
            # 气缸离散化
            df.loc[:, 'cylinder_number'] = pd.cut(df.cylinder_number, bins=[0, 2, 3, 4, 5, 6, 8, 16],
                                                    labels=['2缸', '3缸', '4缸','5缸', '6缸', '8缸', '10缸以上'])
        else:  # 纯电动
            df = data.loc[:, col2]
        # df = df.dropna()
        # df.reindex = range(len(df))

        # 最大功率离散化
        df.loc[:, 'maximum_power'] = pd.cut(df.maximum_power, bins=[ 0, 100, 150, 200, 250, 500, 1000],
                                              labels=['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上'])
        # 上牌时间
        df.loc[:, 'register_time'] = pd.cut(df.register_time, bins=[-1, 1, 3, 5, 8, 50],
                                              labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])
        # 过户次数
        df.loc[:, 'sell_times'] = pd.cut(df.sell_times, bins=[-1, 0.01, 1, 2, 3, 4, 10],
                                           labels=['0次', '1次', '2次', '3次', '4次', '5次及以上'])
        # 车款年份
        df.loc[:, 'model_year'] = pd.cut(df.model_year, bins=[0, 2008, 2013, 2017, 2050],
                                           labels=['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后'])

        return df


    def split_data(self, data):
        '''
        将数据划分为训练集和测试集
        :param data: 数据框
        :return: X_train, X_test, y_train, y_test
        '''

        target = data.iloc[:,-1]  # 目标值
        feature = data.iloc[:,:-1]  # 特征值
        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

        return  X_train, X_test, y_train, y_test, feature, target


    def split_train_(self, data, test_ratio):
        '''
        拆分数据
        :param data: 源数据
        :param test_ratio: 测试集比例
        :return: 训练集和测试集
        '''
        shuffled_indices = np.random.permutation(len(data))    # 打乱序列
        test_set_size = int(len(data) * test_ratio)    # 拆分比例
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        return data.iloc[train_indices], data.iloc[test_indices]


    def get_category(self, car_class, car_brand, car_system):

        # 分类型特征名称
        cols_name_categ_NEV = ['car_brand', 'car_system', 'cylinder_number', 'driving', 'gearbox_type',
                              'intake_form', 'maximum_power', 'register_time', 'sell_times', 'model_year']
        cols_name_categ_EV = ['car_brand', 'car_system', 'driving', 'gearbox_type', 'maximum_power', 'register_time',
                             'sell_times', 'model_year']

        # 分类型特征类别
        cylinder_number = ['2缸', '3缸', '4缸', '5缸', '6缸', '8缸', '10缸以上']
        driving = ['四驱', '后驱', '前驱']
        gearbox_type = ['DCT', 'CVT', 'AMT', 'AT自动', 'MT手动', '超跑变速箱', '单速变速箱']
        intake_form = ['自然吸气', '双增压', '涡轮增压', '机械增压', '四涡轮增压']
        maximum_power = ['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上']
        register_time = ['1年以内', '1-3年', '3-5年', '5-8年', '8年以上']
        sell_times = ['0次', '1次', '2次', '3次', '4次', '5次及以上']
        model_year = ['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后']

        categories_NEV = [cylinder_number, driving, gearbox_type, intake_form, maximum_power, register_time,
                          sell_times, model_year]
        categories_EV = [driving, gearbox_type, maximum_power, register_time, sell_times, model_year]

        if car_class == 'EV':  # 纯电动类型车辆的分类型特征和类别
            categories_EV.insert(0, car_brand)
            categories_EV.insert(1, car_system)
            categories = categories_EV
            cols_categ = cols_name_categ_EV
        else:
            # 非纯电动类型车辆的分类型特征和类别
            categories_NEV.insert(0, car_brand)
            categories_NEV.insert(1, car_system)
            categories = categories_NEV
            cols_categ = cols_name_categ_NEV

        return categories, cols_categ


    def onehot_encode(self, data, categ):
        '''
        One-Hot编码
        :param data: 待编码的分类特征
        :return: One-Hot编码后的数据
        '''
        enc = OneHotEncoder(sparse=False, categories=categ)
        data_encode = enc.fit_transform(data)
        df = pd.DataFrame(data_encode, index=data.index, columns=enc.get_feature_names())
        # print(enc.get_feature_names())
        return df

