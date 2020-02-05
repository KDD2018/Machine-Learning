#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
from selectMySQL import SelectMySQL
from datetime import datetime
from processing import Processing



class GenerateCSV():
    '''
    读取MySQL数据，处理后写入CSV文件
    '''
    def get_car_info_from_customer(self):
        '''
        获取用户提供的车辆信息
        :return: 用户车辆信息字典
        '''
        print('\n客官,请提供您爱车的相关信息：')
        brand_name = input('\n请输入品牌：') or '五菱汽车'
        car_system = input('请输入车系：') or '五菱荣光V'
        car_model_name = input('请输入车型：') or '2018款 1.5L实用型'
        register_time = datetime.strptime(input('请输入上牌时间：') or '2019-03-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（公里）：') or 15500)
        sell_times = float(input('请输入过户次数：') or 0)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                         'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}

        return car_info_from_customer


    def get_feature_classes(self, sql_to_customer_carConfig, sql_to_system, sql_to_level):
        '''
        获取车辆品牌集和车系集
        :param sql_to_customer_carConfig: 查询用户车辆配置
        :param sql_to_brand: 查询品牌集合
        :param sql_to_system: 查询车系集合
        :return: 车辆类型、品牌集、车系集
        '''
        #  获取客户输入车辆信息
        car_info_from_customer = self.get_car_info_from_customer()
        print(f'\n输入完成，当前时间：{datetime.now()}')

        if car_info_from_customer['meter_mile'] > 550000:
            print('\n客观，您的爱车接近报废啦。。。。。')
        else:
            # 查询客户车辆参数配置、类型
            select = SelectMySQL(host='192.168.0.3', user='clean', passwd='Zlpg1234!', db='valuation_web')
            # 根据车辆品牌、车系、车型获取用户车辆参数配置信息
            print('\n正在查询用户车辆信息.....')
            customer_carConfig = select.get_df(sql_to_customer_carConfig.format(car_info_from_customer['car_brand'],
                                                                                car_info_from_customer['car_system'],
                                                                                car_info_from_customer['car_model']))
            car_class=levels=brands=systems = None
            if customer_carConfig:
                # 查询车辆类型
                car_class = customer_carConfig[0]['car_class']
                print(f'\n客官，您的爱车类型属于：{car_class}\n')
                # 查询所有品牌、车系
                print('\n开始获取现有车辆的品牌集和车系集......')
                # car_brand = select.get_df(sql_to_brand.format(car_class))
                # brands = [i['car_brand'] for i in car_brand]
                # print('\n品牌集获取完毕......')
                car_system = select.get_df(sql_to_system.format(car_class))
                systems = [i['car_system'] for i in car_system]
                print('\n车系集获取完毕......')
                car_level = select.get_df(sql_to_level.format(car_class))
                levels = [i['level'] for i in car_level]
                print('\n车辆级别集获取完毕......')
            else:
                print('\n客官，查无此车型')
            return car_class, systems, levels


    def get_car_case(self, sql_to_case_num, car_class, sql_to_CarConfig_CarCase,
                      systems, levels, batch_size=50000):
        '''
        获取车辆案例信息
        :param car_class: 车辆类型  
        :param sql_to_case_num: 查询案例数量
        :param sql_to_CarConfig_CarCase: 查询车辆案例信息及车辆配置信息的SQL
        :param brands: 品牌集
        :param systems: 车系集 
        :param batch_size: 每页案例数量
        :return: 查询案例结果
        '''
        select = SelectMySQL(host='192.168.0.3', user='clean', passwd='Zlpg1234!', db='valuation_web')
        print('\n查询样本数量......')
        num_case_result = select.get_df(sql_to_case_num.format(car_class, car_class, model_year_dict[car_class]))
        len_case = num_case_result[0].get('count(*)')
        print(f'\n样本数量: {len_case}')
        # 查询有几页数据
        epoch = math.ceil(len_case / batch_size)

        # cols_num_list = []
        test = valid = pd.DataFrame()
        for i in range(epoch):
            # 获取案例信息
            print(f'\n开始获取并处理第{i}页案例信息......')
            car_case = select.get_df(sql_to_CarConfig_CarCase.format(car_class, car_class, model_year_dict[car_class],
                                                                     i, batch_size))
            # 将同类车辆案例信息写入DataFrame
            car_case_df = pd.DataFrame(car_case, columns=car_case[0].keys())
            if car_class != 'EV':
                car_case_df.pop('voyage_range')
            #print(car_case_df.isnull().any())
            #print(car_case_df.head())
            #cols_num_list.append(car_case_df.shape[1])

            # 处理数据
            process = Processing()
            categories, col_categ = process.get_category(car_class, systems, levels)
            #print('\n开始预处理...')
            #df_preprocessed = process.preprocess(car_case_df)  # 预处理
            # print(df_preprocessed.isnull().any())
            print('\n开始离散化...')
            df_disrete = process.feature_encode(car_case_df, car_class)  # 离散化
            #print(df_disrete.head())
            # print(df_disrete.isnull().any())
            print('\n开始one-hot编码...')
            df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # one-hot编码
            '''
            df = pd.concat([df_categ, df_preprocessed[['meter_mile', 'vendor_guide_price', 'price']]], axis=1)
            cols_num_list.append(len(df.columns)) 
            '''
            # 拆分数据
            train_data_all, test_data = process.split_train_test(car_case_df, test_ratio=0.2)
            train_data, valid_data = process.split_train_test(train_data_all, test_ratio=0.25)
            test = test.append(test_data, ignore_index=True)  # 测试集
            valid = valid.append(valid_data, ignore_index=True)  # 验证集
            print(f'\n将第{i}页案例信息写入csv......')
            train_data.to_csv(f'/home/kdd/python/car/train/{car_class}/{car_class}_{i}.csv', encoding='utf-8',
                              chunksize=10000, index=False)  # 写入csv
        test.to_csv(f'/home/kdd/python/car/test/{car_class}_test/{car_class}.csv', encoding='utf-8', index=False)
        valid.to_csv(f'/home/kdd/python/car/valid/{car_class}_valid/{car_class}.csv', encoding='utf-8', index=False)
        print('\n**********************CSV文件写入完成*****************************')
        print(f'\n写入完成，当前时间：{datetime.now()}')
        len_train_data = len_case - len(test) - len(valid)

        return len_train_data, len(test), len(valid)


    def run(self):
        '''主程序'''
        len_train = len_test = len_valid = None
        # 1、根据用户输入的车辆信息获取同一类型的车辆案例信息
        car_class, systems, levels = self.get_feature_classes(sql_to_customer_carConfig,
                                                              sql_to_system, sql_to_level)
        print(levels)
        if car_class:
            # 2、对案例信息进行处理并写入csv
            len_train, len_test, len_valid = self.get_car_case(sql_to_case_num, car_class,
                                                               sql_to_CarConfig_CarCase, systems, levels)
            # num_cols = num_cols_list[0]
        return car_class, len_train, len_test, len_valid



# SQL查询语句
sql_to_customer_carConfig = """
SELECT
    displacement,
	# cylinder_number,
	driving,
	maximum_power,
	voyage_range,
	car_class,
	level,
	vendor_guide_price,
	model_year
FROM
	new_car_information_t
WHERE
    car_brand = "{0}"
    AND
        car_system = "{1}"
    AND
	    car_model = "{2}"
"""

sql_to_CarConfig_CarCase = """
SELECT
	# s.car_brand, 
	s.car_system,
	# s.car_model,
	s.displacement,
	s.driving,
	s.maximum_power,
	s.voyage_range,
	# s.car_class,
	s.level,
	s.vendor_guide_price,
	s.model_year,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	s.price
FROM
	second_car_sell_{0} s
	INNER JOIN new_car_information_t n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{1}'
	AND 
	    n.model_year >= '{2}'
	AND 
	    (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))
limit {3}, {4}
"""

sql_to_case_num = """
SELECT
    count(*)
FROM
	second_car_sell_{0} s
	INNER JOIN new_car_information_t n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{1}'
	AND 
	    n.model_year >= '{2}'
	# AND 
	    # (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))
"""

sql_to_brand = """
select
	distinct n.car_brand
from
	new_car_information_t n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
order by n.id
"""

sql_to_system = """
select
	distinct n.car_system
from
	new_car_information_t n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
order by n.id
"""

sql_to_level = """
select
	distinct n.level
from
	new_car_information_t n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
order by n.id
"""





# 用于过滤太老旧的车
model_year_dict = {'saloon': 2005, 'suv': 2007, 'mpv': 2006, 'minibus': 2007, 'supercar': 0, 'EV': 0}

pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    generate = GenerateCSV()
    car_class, len_train, len_test, len_valid = generate.run()

