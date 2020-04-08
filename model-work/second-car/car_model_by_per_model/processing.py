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

    def feature_encode(self, data, car_class):
        '''
        特征离散编码
        :param data: 数据框
        :param col1: EV特征
        :param col1: not EV特征
        :return: 离散化处理的数据框
        '''
        # 非纯电动特征名称 'normalized_vendor_guide_price',
        features_for_NEV = ['car_model', 'displacement', 'driving', 'maximum_power', 'car_age',
                            'mile_per_year', 'mileage_newness_rate', 'sell_times',  'model_year', 'hedge_ratio']
        # 纯电动特征名称
        features_for_EV = ['car_model', 'driving', 'maximum_power', 'voyage_range',  'car_age', 'mile_per_year',
                           'mileage_newness_rate', 'sell_times', 'model_year', 'hedge_ratio']
        if car_class == 'EV':
            df = data.loc[:, features_for_EV]
            #df.loc[:, 'voyage_range'] = pd.cut(df['voyage_range'], bins=[0, 100, 250, 350, 450, 600, 1000],
            #                                   labels=['100公里以内', '100-250公里', '250-350公里', '350-450公里',
            #                                          '450-600公里', '600公里以上'])
            df.loc[:, 'voyage_range'] = pd.cut(df['voyage_range'], bins=[0, 100, 180, 250, 350, 450, 1000],
                                               labels=['100公里以内', '100-180公里', '180-250公里', '250-350公里',
                                                        '350-450公里','450公里以上'])
        else:
            df = data.loc[:, features_for_NEV]
            # 排量离散化
            df.loc[:, 'displacement'] = pd.cut(df.displacement, bins=[0, 1.0, 1.6, 2.5, 4, 6, 8],
                                               labels=['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上'])
        # print(data.isnull().any())
        # 最大功率离散化
        df.loc[:, 'maximum_power'] = pd.cut(df.maximum_power, bins=[ 0, 100, 150, 200, 250, 500, 5000],
                                            labels=['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW',
                                                    '500KW以上'])
        # 过户次数
        df.loc[:, 'sell_times'] = pd.cut(df.sell_times, bins=[-1, 0.01, 1, 2, 3, 20],
                                         labels=['0次', '1次', '2次', '3次', '4次及以上'])  # 左开右闭
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


    def get_category(self, car_class,  car_model):
        '''
        获取分类类别
        :param car_class: 车辆类型 car_brand, car_system,
        :param car_brand: 品牌集
        :param car_system: 车系集
        :return: 分类及相应特征
        '''
        # 分类型特征名称 'car_brand', 'car_system',
        categorical_features_for_NEV = ['car_model', 'displacement', 'driving', 'maximum_power',
                                         'sell_times', 'model_year']
        categorical_features_for_EV = ['car_model', 'driving',  'maximum_power',
                                       'sell_times', 'model_year', 'voyage_range']
        # 分类型特征类别
        driving = ['四驱', '后驱', '前驱']
        maximum_power = ['100KW以内', '100-150KW', '150-200KW', '200-250KW', '250-500KW', '500KW以上']
        displacement = ['0-1.0L', '1.0-1.6L', '1.6-2.5L', '2.5-4L', '4-6L', '6L以上']
        sell_times = ['0次', '1次', '2次', '3次', '4次及以上']
        model_year = ['2008款以前', '2009-2012款', '2013-2017款', '2018款及以后']
        #voyage_range = ['100公里以内', '100-250公里', '250-350公里', '350-450公里', '450-600公里', '600公里以上']
        voyage_range = ['100公里以内', '100-180公里', '180-250公里', '250-350公里','350-450公里','450公里以上']
        categories_NEV = [displacement, driving, maximum_power, sell_times, model_year]
        categories_EV = [driving, maximum_power, sell_times, model_year, voyage_range]

        if car_class == 'EV':  # 纯电动类型车辆的分类型特征和类别
            if len(car_model) > 1:
                categories_EV.insert(0, car_model)
            else:
                categorical_features_for_EV.remove('car_model')
            #categories_EV.insert(1, car_system)
            categories = categories_EV
            categorical_features = categorical_features_for_EV
        else:
            # 非纯电动类型车辆的分类型特征和类别
            #categories_NEV.insert(0, car_brand)
            #categories_NEV.insert(1, car_system)
            if len(car_model) > 1:
                categories_NEV.insert(0, car_model)
            else:
                categorical_features_for_NEV.remove('car_model')
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

        # 获取分类型特征及特征类别
        categories, categorical_features = self.get_category(car_class, car_models)
        #print(categorical_features)
        # 分类型特征离散化
        df_disrete = self.feature_encode(data, car_class)
        #print(df_disrete.isnull().any())
        # one-hot编码
        df_categ = self.onehot_encode(df_disrete[categorical_features], categories)
        df = pd.concat([df_categ,
                        df_disrete[['car_age', 'mile_per_year', 'mileage_newness_rate', 'hedge_ratio']]], axis=1)

        return df

