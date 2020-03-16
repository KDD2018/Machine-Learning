#!/usr/bin/python3
# -*- coding: utf-8 -*-



from tensorflow import keras
import pandas as pd
from datetime import datetime
from processing import Processing
import numpy as np
import pymysql


class LoadModelPredict(object):
    '''加载模型进行预测'''
    def __init__(self):
        self.default_conn = pymysql.connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!',
                                            db='valuation_web', port=3306, charset='utf8',
                                            cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.default_cursor = self.default_conn.cursor()


    def get_car_info_from_customer(self):
        '''
        获取用户提供的车辆信息
        :return: 用户车辆表
        '''

        brand_name = input('请输入品牌：') or '宝马'
        car_system = input('请输入车系：') or '宝马i3'
        car_model_name = input('请输入车型：') or '2015款 时尚型'
        register_time = datetime.strptime(input('请输入上牌时间：') or '2016-04-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（万公里）：') or 2.18)
        sell_times = float(input('请输入过户次数：') or 0)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                                  'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}
        car_info_from_customer = pd.DataFrame([car_info_from_customer])
        return car_info_from_customer


    def get_test_case(self):
        '''
        获取待预测样本集
        :param test_case_path: 待预测样本路径
        :return: 
        '''

        try:
            button = int(input('\n客官, 您是要单样本预测还是多样本批量预测(单样本:1,多样本:0): ') or 1)

            if button == 1:
                # 单样本预测
                test_case = self.get_car_info_from_customer()
                return test_case, button
            elif button == 0:
                # 多样本批量预测
                test_case = pd.read_csv('../second_car_sell.csv')
                test_case['register_time'] = pd.to_datetime(test_case['register_time'])
                return test_case, button
        except ValueError as ex:
            print('\n请您输入正确信息!!!')


    def sql_to_parameter(self, car_class):
        '''
        查询用于数据处理的参数
        :param car_class: 车辆类型
        :return: 参数字典
        '''
        sql_to_params = """ select * from second_car_params where car_class='{0}'"""
        self.default_cursor.execute(sql_to_params.format(car_class))
        params = self.default_cursor.fetchall()[0]
        return params


    def preprocess(self, df, car_class):
        '''
        预处理
        :param df: 
        :return: 
        '''
        params = self.sql_to_parameter(car_class)
        #print(df)
        df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'] / 10000.0  # 换算成万元
        df.loc[:, 'mileage_newness_rate'] = 1 - df['meter_mile'] / 60  # 转换成行驶里程成新率
        df.loc[:, 'car_age'] = df['register_time'].map(
            lambda x: ((datetime.now().year - x.year) * 12 + (datetime.now().month - x.month)) / 12)
        #print(df)
        #df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'].map(lambda x: np.log1p(x))
        #df.loc[:, 'vendor_guide_price'] = df['vendor_guide_price'].map(
        #    lambda x: (x - float(params['log_vendor_guide_price_min']))/(
        #        float(params['log_vendor_guide_price_max']) - float(params['log_vendor_guide_price_min'])))
        df.loc[:, 'mile_per_year'] = df.loc[:, 'meter_mile'] / df.loc[:, 'car_age']
        if car_class == 'EV':
            df.loc[:, 'voyage_range'] = df['voyage_range'] / 1000.0  # 换算成公里
        else:
            df.pop('voyage_range')
        del df['car_class'], df['car_model'], df['meter_mile']

        return df, params


    def load_model_predict(self, model_path, x):
        '''
        加载模型进行评估
        :param path: 模型路径
        :param test_data: 测试数据集路径
        :return: 预测值
        '''
        new_model = keras.models.load_model(model_path)
        y_hat = new_model.predict(np.array(x))

        return y_hat


    def run(self):

        # 加载待预测样本信息
        test_case_all, button = self.get_test_case()
        if button == 0:
            # 多样本
            test_case = test_case_all.drop(columns=['price'])
        else:
            # 单样本
            test_case = test_case_all

        sql_to_customer_carConfig = """SELECT displacement, driving, maximum_power, voyage_range, car_class, level,
                                       vendor_guide_price, model_year FROM new_car_information_t WHERE car_brand = '{0}' 
                                       AND car_system = '{1}' AND car_model = '{2}'"""
        sql_to_brand = """SELECT DISTINCT s.car_brand FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON
                          s.car_model_id = n.car_model_id WHERE s.isdeleted=0 ORDER BY n.id"""
        sql_to_system = """SELECT DISTINCT s.car_system FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                           s.car_model_id = n.car_model_id WHERE s.isdeleted=0 ORDER BY n.id"""
        sql_to_level = """SELECT DISTINCT s.level FROM second_car_sell_{0} s INNER JOIN new_car_information_t n ON 
                          s.car_model_id = n.car_model_id WHERE s.isdeleted=0 ORDER BY n.id"""

        if test_case.shape[0]:
            for i in range(test_case.shape[0]):

                customer_car_info = test_case.loc[i, :].to_dict()

                if customer_car_info['meter_mile'] > 59:
                    print('\n客观，您的爱车接近报废啦。。。。。')
                    test_case_all.loc[i, 'predicted_price'] = 0
                    continue
                else:
                    # 查询客户车辆参数配置、类型
                    self.default_cursor.execute(sql_to_customer_carConfig.format(customer_car_info['car_brand'],
                                                                                 customer_car_info['car_system'],
                                                                                 customer_car_info['car_model']))
                    customer_carConfig = self.default_cursor.fetchall()
                    if customer_carConfig:
                        # 合并客户车辆配置信息和用户输入信息
                        customer_car_info = dict(customer_car_info, **customer_carConfig[0])
                        # 客户汽车所属类别
                        car_class = customer_car_info['car_class']
                        print(f'\n客官，您的爱车类型属于：{car_class}\n')

                        if car_class not in ['saloon', 'suv', 'mpv', 'supercar', 'minibus', 'EV']:
                            print('\n客官，现不支持此类车型的估值...')
                            test_case_all.loc[i, 'predicted_price'] = 0
                            continue
                        else:
                            # 将客户车辆信息写入DataFrame
                            customer_car_df = pd.DataFrame([customer_car_info])
                            # 查询所有品牌、车系、车辆级别
                            self.default_cursor.execute(sql_to_level.format(car_class))
                            car_level = self.default_cursor.fetchall()
                            self.default_cursor.execute(sql_to_brand.format(car_class))
                            car_brand = self.default_cursor.fetchall()
                            self.default_cursor.execute(sql_to_system.format(car_class))
                            car_system = self.default_cursor.fetchall()
                            brands = [i['car_brand'] for i in car_brand]
                            systems = [i['car_system'] for i in car_system]
                            levels = [i['level'] for i in car_level]
                            try:
                                # 2、对用户车信息进行处理
                                customer_car_df, params =self.preprocess(customer_car_df, car_class) # 预处理
                                #print(customer_car_df)
                                process = Processing()
                                categories, categorical_features = process.get_category(car_class,  levels)
                                df_disrete = process.feature_encode(customer_car_df, car_class)  # 离散化
                                #print(df_disrete.isnull().any())
                                #print(df_disrete[df_disrete['voyage_range'].isnull()])
                                #print(df_disrete)

                                df_categ = process.onehot_encode(df_disrete[categorical_features], categories)  # one-hot编码
                                df = pd.concat([df_categ, customer_car_df[['car_age', 'mile_per_year', 'mileage_newness_rate']]], axis=1) #
                                df = df.astype('float32')
                                #print(df)

                                # 3、预测用户车辆价值
                                model_dir = f'../../model-param/{car_class}/{car_class}.h5'  # 模型路径
                                # 加载预测
                                y_hat = self.load_model_predict(model_path=model_dir, x=df)[0][0]
                                #y_hat = y_hat * (float(params['log_price_max']) - float(params['log_price_min'])) + \
                                        #float(params['log_price_min'])
                                print(y_hat)
                                test_case_all.loc[i, 'predicted_price'] = customer_car_df.loc[0, 'vendor_guide_price'] * y_hat
                                #test_case_all.loc[i, 'predicted_price'] = np.expm1(y_hat[0][0])
                                print(test_case_all.loc[i, 'predicted_price'])
                            except ValueError as ex:
                                print(ex)
                                test_case_all.loc[i, 'predicted_price'] = 0
                                continue
                    else:
                        print('\n客观, 经查询无此车型!!!')
                        test_case_all.loc[i, 'predicted_price'] = 0
                        continue
            test_case_all.to_csv('./test_case_all.csv', encoding='utf-8')
        else:
            print('\n客官, 未能正确获取待测样本!!!')



pd.set_option('display.max_columns', None)



if __name__ == '__main__':

    model = LoadModelPredict()
    model.run()

