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


def csv_read_dataset(filename_list, features_num, batch_size, n_readers=5,
                     n_parse_threads=5, shuffle_buffer_size=20000):
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
    data_set = data_set.map(lambda row: parse_csv_line(line=row, n_fields=features_num),
                            num_parallel_calls=n_parse_threads)
    data_set = data_set.batch(batch_size=batch_size)


    return data_set


def plot_learning_curves(history):
    '''
    绘制学习曲线
    :param history: 训练过程
    :return: None
    '''
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.show()


def ridge_model(features_num, train_data, valid_data, len_train,
                len_valid, test_data, len_test, batch_size):
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
    '''
    input1 = keras.layers.Input(shape=(features_num-4, ))
    input2 = keras.layers.Input(shape=(3, ))
    hidden1 = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))(input1)
    hidden2_1 = keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(input2)

    concat = keras.layers.concatenate([hidden1, hidden2_1])
    outputs = keras.layers.Dense(1)(concat)

    model = keras.models.Model(inputs=[input1, input2], outputs=[outputs])
    '''
    model = keras.models.Sequential([
        keras.layers.Dense(1,  input_shape=(features_num-1, ),
                           kernel_regularizer=keras.regularizers.l2(0.01)),
    ])
    # 模型概要
    model.summary()
    # 编译模型
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # 训练模型
    callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
    history = model.fit(train_data,
                        validation_data=valid_data,
                        steps_per_epoch=len_train // batch_size,
                        validation_steps=len_valid // batch_size, epochs=100, callbacks=callbacks)
    # 绘制学习曲线
    plot_learning_curves(history)
    # 模型评估
    print('\n模型在测试集上的表现：')
    model.evaluate(test_data, steps=len_test // batch_size)

    return model


def run():
    # 1、读取MySQL、处理、写入CSV文件
    generate = GenerateCSV()
    car_class, len_train, len_test, len_valid, features_num = generate.run()

    if features_num:
        #
        batch_size = 256

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
        train_ds = csv_read_dataset(train_filename_list, features_num=features_num, batch_size=batch_size)
        valid_ds = csv_read_dataset(valid_filename_list, features_num=features_num, batch_size=batch_size)
        test_ds = csv_read_dataset(test_filename_list, features_num=features_num, batch_size=batch_size)

        # 训练模型
        model = ridge_model(features_num, train_ds, valid_ds, len_train,
                            len_valid, test_ds, len_test, batch_size)
        # 保存模型
        model.save(model_dir)


if __name__ == '__main__':
    print(f'\n开始执行，当前时间: {datetime.now()}')
    run()
    print(f'\n模型生成完毕，当前时间：{datetime.now()}')


