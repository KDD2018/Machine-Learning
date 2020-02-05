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
        if data.loc[0, 'car_class'] == 'EV':
            del data['cylinder_number'], data['intake_form'], data['level']
            #data.loc[:, 'voyage_range'] = data['voyage_range'] / 1000.0  # 换算成公里
        #else:
         #   del data['voyage_range']
        data.dropna(inplace=True)
        data.index = range(len(data))
        #data.loc[:, 'meter_mile'] = data['meter_mile'] / 10000.0  # 换算成万公里
        #data.loc[:, 'price'] = data['price'] / 10000.0  # 换算成万元
        #data.loc[:, 'vendor_guide_price'] = data['vendor_guide_price'] / 10000.0  # 换算成万元
        #df_ = data.loc[data.meter_mile < 55, :].copy()  # 过滤掉40万公里以上的案例
        #df = df_.loc[df_.sell_times <= 10, :].copy()  # 过滤掉过户次数10次以上的
        #df.loc[:, 'meter_mile'] = 1 - df['meter_mile'] / 60  # 转换成行驶里程成新率
        #df.loc[:, 'register_time'] = df['register_time'].map(
            #lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
        df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'].map(lambda x: np.log1p(x))
        df.loc[:, 'price'] = df['price'].map(lambda x: np.log1p(x))

        return df


    def feature_encode(self, data, car_class):
        '''
        特征离散编码
        :param data: 数据框
        :param col1: EV特征
        :param col1: not EV特征
        :return: 离散化处理的数据框
        '''
        # 非纯电动特征名称
        features_for_NEV = ['car_system', 'level', 'displacement', 'driving', 'maximum_power', 'register_time',
                            'meter_mile', 'sell_times', 'vendor_guide_price', 'model_year', 'price']
        # 纯电动特征名称
        features_for_EV = ['car_system', 'driving', 'maximum_power', 'voyage_range', 'register_time', 'meter_mile',
                           'sell_times', 'vendor_guide_price', 'model_year', 'price']
        if car_class == 'EV':
            df = data.loc[:, features_for_EV]
            df.loc[:, 'voyage_range'] = pd.cut(df['voyage_range'], bins=[0, 0.1, 0.25, 0.35, 0.45, 0.6, 1],
                                               labels=['100公里以内', '100-250公里', '250-350公里', '350-450公里',
                                                       '450-600公里', '600公里以上'])
        else:
            df = data.loc[:, features_for_NEV]
            # 排量离散化
            df.loc[:, 'displacement'] = pd.cut(df.displacement, bins=[0, 1.0, 1.6, 2.5, 4, 6, 8],
                                               labels=['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上'])
        # print(data.isnull().any())
        # 最大功率离散化
        df.loc[:, 'maximum_power'] = pd.cut(df.maximum_power, bins=[ 0, 100, 150, 200, 250, 500, 5000],
                                              labels=['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上'])
        # 上牌时间
        df.loc[:, 'register_time'] = pd.cut(df.register_time, bins=[-1, 1, 3, 5, 8, 50],
                                              labels=['1年以内', '1-3年', '3-5年', '5-8年', '8年以上'])
        # 过户次数
        df.loc[:, 'sell_times'] = pd.cut(df.sell_times, bins=[-1, 0.01, 1, 2, 3, 4, 20],
                                           labels=['0次', '1次', '2次', '3次', '4次', '5次及以上'])
        # 车款年份
        df.loc[:, 'model_year'] = pd.cut(df.model_year, bins=[0, 2008, 2013, 2017, 2100],
                                           labels=['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后'])

        return df


    def split_train_test(self, data, test_ratio):
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


    def get_category(self, car_class, car_system, car_level):
        '''
        获取分类类别
        :param car_class: 车辆类型
        :param car_brand: 品牌集
        :param car_system: 车系集
        :return: 分类及相应特征
        '''
        # 分类型特征名称
        categorical_features_for_NEV = ['car_system', 'level','displacement', 'driving', 'maximum_power',
                                       'register_time', 'sell_times', 'model_year']
        categorical_features_for_EV = ['car_system', 'driving',  'maximum_power', 'register_time', 'sell_times',
                                      'model_year', 'voyage_range']
        # 分类型特征类别
        driving = ['四驱', '后驱', '前驱']
        maximum_power = ['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上']
        register_time = ['1年以内', '1-3年', '3-5年', '5-8年', '8年以上']
        displacement = ['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上']
        sell_times = ['0次', '1次', '2次', '3次', '4次', '5次及以上']
        model_year = ['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后']
        voyage_range = ['100公里以内', '100-250公里', '250-350公里', '350-450公里', '450-600公里', '600公里以上']
        categories_NEV = [displacement, driving, maximum_power, register_time,sell_times, model_year]
        categories_EV = [driving, maximum_power, register_time, sell_times, model_year, voyage_range]

        if car_class == 'EV':  # 纯电动类型车辆的分类型特征和类别
            # categories_EV.insert(0, car_brand)
            categories_EV.insert(0, car_system)
            categories = categories_EV
            categorical_features = categorical_features_for_EV
        else:
            # 非纯电动类型车辆的分类型特征和类别
            # categories_NEV.insert(0, car_brand)
            categories_NEV.insert(0, car_system)
            categories_NEV.insert(1, car_level)
            categories = categories_NEV
            categorical_features = categorical_features_for_NEV

        return categories, categorical_features


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

