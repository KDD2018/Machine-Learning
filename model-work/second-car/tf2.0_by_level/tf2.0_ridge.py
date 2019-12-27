#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os
from tensorflow import keras
from generate_csv import GenerateCSV
import matplotlib.pyplot as plt




def get_file_list(path):
    '''
    返回文件列表
    :param path: 文件所在路径
    :return: 文件列表
    '''
    file_name = os.listdir(path)
    file_list = [os.path.join(path, file) for file in file_name]

    return file_list


def parse_csv_line(line, n_fields):
    '''
    解析csv每行内容
    :param line: csv每行内容
    :param n_fields: CSV字段数
    :return: 解析后的内容
    '''
    records = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, use_quote_delim=True, record_defaults=records)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1])
    
    return x, y    


def csv_read_dataset(filename_list, num_cols, n_readers=5, n_parse_threads=5, shuffle_buffer_size=50000):
    '''
    读取并解析CSV
    :param filename_list: CSV文件名列表
    :param batch_size: 批大小
    :param num_cols: 字段数
    :param n_readers: 读取并行度
    :param n_parse_threads: 解析并行度
    :param shuffle_buffer_size: 混淆度
    :return: 批数据
    '''
    filename_ds = tf.data.Dataset.list_files(filename_list).repeat()
    data_set =  filename_ds.interleave(lambda filename: tf.data.TextLineDataset(filename).skip(1),
                                       cycle_length=n_readers)
    data_set.shuffle(shuffle_buffer_size)
    data_set = data_set.map(lambda row: parse_csv_line(line=row, n_fields=num_cols),
                            num_parallel_calls=n_parse_threads)
    data_set = data_set.batch(batch_size=64)


    return data_set


def plot_learning_curves(history):
    '''
    绘制学习曲线
    :param history: 
    :return: 
    '''
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.show()


def ridge_model(num_cols, train_data, valid_data, len_train, len_valid, test_data, len_test):
    '''
    创建模型
    :param num_cols: 特征数
    :param train_data: 训练集
    :param valid_data: 验证集
    :param len_train: 训练集样本数量
    :param len_valid: 验证集样本数量
    :param batch_size: 批大小
    :return: model
    '''
    # 构造模型
    model = keras.models.Sequential([
        keras.layers.Dense(1,  input_shape=(num_cols-1, ), kernel_regularizer=keras.regularizers.l2(0.01)),
        # keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.01)),
        # keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.01)),
        # keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))
    ])
    # 模型概要
    model.summary()
    # 编译模型
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # 训练模型
    callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
    history = model.fit(train_data, validation_data=valid_data, steps_per_epoch=len_train // 64,
                        validation_steps=len_valid // 64, epochs=10, callbacks=callbacks)
    plot_learning_curves(history)
    # 模型评估
    print('\n模型在测试集上的表现：')
    model.evaluate(test_data, steps=len_test // 64)

    return model


def run():
    # 1、读取MySQL、处理、写入CSV文件
    generate = GenerateCSV()
    num_cols, car_class, car_level, len_train, len_test, len_valid = generate.run()

    # num_cols = 263
    # car_class = 'supercar'
    # len_train = 9209
    # len_test = 3069
    # len_valid = 3069


    if num_cols:
        if car_class in ['saloon', 'suv']:
            # 数据路径
            train_data_dir = f'/home/kdd/python/car/train/{car_class}/{car_level}'
            test_data_dir = f'/home/kdd/python/car/test/{car_class}_test/{car_level}'
            valid_data_dir = f'/home/kdd/python/car/valid/{car_class}_valid/{car_level}'
            # 模型路径
            model_dir = f'../../model-param/{car_class}/{car_class}_{car_level}.h5'
        else:
            # 数据路径
            train_data_dir = f'/home/kdd/python/car/train/{car_class}'
            test_data_dir = f'/home/kdd/python/car/test/{car_class}_test/'
            valid_data_dir = f'/home/kdd/python/car/valid/{car_class}_valid/'
            # 模型路径
            model_dir = f'../../model-param/{car_class}/{car_class}.h5'

        # 读取csv文件并解析
        train_filename_list = get_file_list(path=train_data_dir)
        valid_filename_list = get_file_list(path=valid_data_dir)
        test_filename_list = get_file_list(path=test_data_dir)
        train_data = csv_read_dataset(train_filename_list, num_cols=num_cols)
        valid_data = csv_read_dataset(valid_filename_list, num_cols=num_cols)
        test_data = csv_read_dataset(test_filename_list, num_cols=num_cols)

        # 训练模型
        model = ridge_model(num_cols, train_data, valid_data, len_train, len_valid, test_data, len_test)
        # 保存模型
        model.save(model_dir)


if __name__ == '__main__':
    print(f'\n开始执行，当前时间: {datetime.now()}')
    run()
    print(f'\n模型生成完毕，当前时间：{datetime.now()}')


