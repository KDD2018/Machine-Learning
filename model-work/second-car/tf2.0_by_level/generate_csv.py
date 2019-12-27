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
        brand_name = input('\n请输入品牌：') or '宝马'
        car_system = input('请输入车系：') or '宝马M3'
        car_model_name = input('请输入车型：') or '2018款 M3四门轿车'
        register_time = datetime.strptime(input('请输入上牌时间：') or '2019-01-01', '%Y-%m-%d')
        meter_mile = float(input('请输入已行驶里程（公里）：') or 10000)
        sell_times = float(input('请输入过户次数：') or 0)
        car_info_from_customer = {'car_brand': brand_name, 'car_system': car_system, 'car_model': car_model_name,
                         'register_time': register_time, 'meter_mile': meter_mile, 'sell_times': sell_times}


        return car_info_from_customer


    def get_brands_and_systems(self, sql_to_customer_carConfig, sql_to_brand, sql_to_system, sql_to_brand_of_saloon_suv,
                               sql_to_system_of_saloon_suv):
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
            car_class=car_level=brands=systems = None
            if customer_carConfig:
                # 查询车辆类型
                car_class = customer_carConfig[0]['car_class']
                car_level = customer_carConfig[0]['level']
                print(f'\n客官，您的爱车类型属于：{car_class}, 车辆级别属于{car_level}\n')

                if car_class in ['saloon', 'suv']:
                    # 查询所有品牌、车系
                    print('\n开始获取现有车辆的品牌集和车系集......')
                    car_brand = select.get_df(sql_to_brand_of_saloon_suv.format(car_class, car_level))
                    brands = [i['car_brand'] for i in car_brand]
                    print('\n品牌集获取完毕......')
                    car_system = select.get_df(sql_to_system_of_saloon_suv.format(car_class, car_level))
                    systems = [i['car_system'] for i in car_system]
                    print('\n车系集获取完毕......')
                else:
                    # 查询所有品牌、车系
                    print('\n开始获取现有车辆的品牌集和车系集......')
                    car_brand = select.get_df(sql_to_brand.format(car_class))
                    brands = [i['car_brand'] for i in car_brand]
                    print('\n品牌集获取完毕......')
                    car_system = select.get_df(sql_to_system.format(car_class))
                    systems = [i['car_system'] for i in car_system]
                    print('\n车系集获取完毕......')
            else:
                print('\n客官，查无此车型')
            return car_class, car_level, brands, systems


    def get_car_case(self, car_class, car_level, sql_to_case_num_of_saloon_suv, sql_to_case_num,
                     sql_to_CarConfig_CarCase_of_saloon_suv, sql_to_CarConfig_CarCase, brands, systems, batch_size=50000):
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
        if car_class in ['saloon', 'suv']:
            num_case_result = select.get_df(sql_to_case_num_of_saloon_suv.format(car_class, car_level,
                                                                                 model_year_dict[car_class]))
        else:
            num_case_result = select.get_df(sql_to_case_num.format(car_class, model_year_dict[car_class]))
        len_case = num_case_result[0].get('count(*)')

        # 查询有几页数据
        epoch = math.ceil(len_case / batch_size)

        cols_num_list = []
        test = valid = pd.DataFrame()
        if car_class in ['saloon', 'suv']:
            for i in range(epoch):
                # 获取案例信息
                print(f'\n开始获取并处理第{i}页案例信息......')
                car_case = select.get_df(sql_to_CarConfig_CarCase_of_saloon_suv.format(car_class, car_level,
                                                                                       model_year_dict[car_class],
                                                                                       i, batch_size))
                # 将同类车辆案例信息写入DataFrame
                car_case_df = pd.DataFrame(car_case, columns=car_case[0].keys())
                # 处理数据
                process = Processing()
                categories, col_categ = process.get_category(car_class, brands, systems)
                print('\n开始预处理...')
                df_preprocessed = process.preprocess(car_case_df)  # 预处理
                print('\n开始离散化....')
                df_disrete = process.feature_encode(df_preprocessed, col_NEV, col_EV)  # 离散化
                print('\n开始one-hot编码')
                df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # onehot编码

                df = pd.concat([df_categ, df_preprocessed[['meter_mile', 'vendor_guide_price', 'price']]], axis=1)
                cols_num_list.append(len(df.columns))
                # 拆分数据
                train_data_all, test_data = process.split_train_test(df, test_ratio=0.2)
                train_data, valid_data = process.split_train_test(train_data_all, test_ratio=0.25)
                test = test.append(test_data, ignore_index=True)  # 测试集
                valid = valid.append(valid_data, ignore_index=True)  # 验证集
                print(f'\n将第{i}页案例信息写入csv......')
                train_data.to_csv(f'/home/kdd/python/car/train/{car_class}/{car_level}/{car_class}_{i}.csv',
                                  encoding='utf-8', chunksize=10000, index=False)  # 写入csv
            test.to_csv(f'/home/kdd/python/car/test/{car_class}_test/{car_level}/{car_class}.csv', encoding='utf-8', index=False)
            valid.to_csv(f'/home/kdd/python/car/valid/{car_class}_valid/{car_level}/{car_class}.csv', encoding='utf-8', index=False)
        else:
            for i in range(epoch):
                # 获取案例信息
                print(f'\n开始获取并处理第{i}页案例信息......')
                car_case = select.get_df(sql_to_CarConfig_CarCase.format(car_class, model_year_dict[car_class],
                                                                         i, batch_size))
                # 将同类车辆案例信息写入DataFrame
                car_case_df = pd.DataFrame(car_case, columns=car_case[0].keys())
                # 处理数据
                process = Processing()
                categories, col_categ = process.get_category(car_class, brands, systems)
                print('\n开始预处理...')
                df_preprocessed = process.preprocess(car_case_df)  # 预处理
                print('\n开始离散化....')
                df_disrete = process.feature_encode(df_preprocessed, col_NEV, col_EV)  # 离散化
                print('\n开始one-hot编码')
                df_categ = process.onehot_encode(df_disrete[col_categ], categories)  # onehot编码

                df = pd.concat([df_categ, df_preprocessed[['meter_mile', 'vendor_guide_price', 'price']]], axis=1)
                cols_num_list.append(len(df.columns))
                # 拆分数据
                train_data_all, test_data = process.split_train_test(df, test_ratio=0.2)
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

        return cols_num_list, len_train_data, len(test), len(valid)


    def run(self):

        num_cols = len_train = len_test = len_valid = None
        # 1、根据用户输入的车辆信息获取同一类型的车辆案例信息
        car_class, car_level, brands, systems = self.get_brands_and_systems(sql_to_customer_carConfig, sql_to_brand,
                                                                            sql_to_system, sql_to_brand_of_saloon_suv,
                                                                            sql_to_system_of_saloon_suv)
        if car_class:
            # 2、对案例信息进行处理并写入csv
            num_cols_list, len_train, len_test, len_valid = self.get_car_case(car_class, car_level, sql_to_case_num_of_saloon_suv,
                                                                              sql_to_case_num, sql_to_CarConfig_CarCase_of_saloon_suv,
                                                                              sql_to_CarConfig_CarCase, brands, systems)
            num_cols = num_cols_list[0]
        return num_cols, car_class, car_level, len_train, len_test, len_valid



# SQL查询语句
sql_to_customer_carConfig = """
SELECT
	cylinder_number,
	driving,
	gearbox_type,
	intake_form,
	maximum_power,
	voyage_range,
	car_class,
	level,
	vendor_guide_price,
	model_year
FROM
	new_car_information
WHERE
    car_brand = "{0}"
    AND
        car_system = "{1}"
    AND
	    car_model = "{2}"
"""

sql_to_CarConfig_CarCase = """
SELECT
	n.car_brand, 
	n.car_system,
	n.car_model,
	n.cylinder_number,
	n.driving,
	n.gearbox_type,
	n.intake_form,
	n.maximum_power,
	n.voyage_range,
	n.car_class,
	n.vendor_guide_price,
	n.model_year,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{0}'
	AND 
	    n.model_year >= '{1}'
	AND 
	    (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))

limit {2}, {3}
"""

sql_to_CarConfig_CarCase_of_saloon_suv = """
SELECT
	n.car_brand, 
	n.car_system,
	n.car_model,
	n.cylinder_number,
	n.driving,
	n.gearbox_type,
	n.intake_form,
	n.maximum_power,
	n.voyage_range,
	n.car_class,
	n.vendor_guide_price,
	n.model_year,
	s.register_time,
	s.meter_mile,
	s.sell_times,
	s.price
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{0}'
	AND
	    n.level = '{1}'
	AND 
	    n.model_year >= '{2}'
	AND 
	    (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))

LIMIT {3}, {4}
"""

sql_to_case_num = """
SELECT
    count(*)
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{0}'
	# AND
	#     n.level = '{1}'
	AND 
	    n.model_year >= '{1}'
	# AND 
	    # (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))
"""

sql_to_case_num_of_saloon_suv = """
SELECT
    count(*)
FROM
	second_car_sell s
	INNER JOIN new_car_information n ON s.car_model_id = n.car_model_id
WHERE
	n.car_class = '{0}'
	AND
	    n.level = '{1}'
	AND 
	    n.model_year >= '{2}'
	# AND 
	    # (s.createTime >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR))
"""

sql_to_brand = """
select
	distinct n.car_brand
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
order by n.id
"""

sql_to_brand_of_saloon_suv = """
select
	distinct n.car_brand
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
	AND
	    n.level = '{1}'
order by n.id
"""

sql_to_system = """
select
	distinct n.car_system
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
order by n.id
"""

sql_to_system_of_saloon_suv = """
select
	distinct n.car_system
from
	new_car_information n
	INNER JOIN second_car_sell s ON s.car_model_id = n.car_model_id
where
	n.car_class = '{0}'
	AND
	    n.level = '{1}'
order by n.id
"""


# 非纯电动特征名称
col_NEV = ['car_brand', 'car_system', 'cylinder_number', 'driving', 'gearbox_type', 'intake_form',
           'maximum_power', 'register_time', 'meter_mile', 'sell_times', 'vendor_guide_price',
           'model_year', 'price']
# 纯电动特征名称
col_EV = ['car_brand', 'car_system', 'driving', 'gearbox_type','maximum_power', 'voyage_range', 'register_time',
          'meter_mile', 'sell_times', 'vendor_guide_price', 'model_year', 'price']


# 用于过滤太老旧的车
model_year_dict = {'saloon': 2005, 'suv': 2007, 'mpv': 2006, 'minibus': 2007, 'supercar': 0, 'EV': 0}

pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    generate = GenerateCSV()
    num_cols, car_class, car_level, len_train, len_test, len_valid = generate.run()

