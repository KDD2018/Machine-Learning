#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import numpy as np


def weight_init(shape):
    '''
    初始化权重
    :param shape: 权重的形状 
    :return: 初始化的权重值
    '''
    weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0), name='weight')

    return weight


def bias_init(shape):
    '''
    初始化偏置
    :param shape: 偏置的形状 
    :return: 初始化的偏置值
    '''
    bias = tf.Variable(tf.constant(0.0, shape=shape))

    return bias


def read_csv(filelist):
    '''
    批获取csv数据
    :param filelist: 文件列表
    :return: 
    '''
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、 构造CSV阅读器，读取队列数据
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
    # records = [['None'], ['None'], ['None'], [0], ['None'], ['None'], ['None'], [1.0], [1.0], ['None'], [1.0], [0],
    #            ['None'], [1.0], [1.0], [1.0]]
    records = [[0.0] for _ in range(679)]
    data = tf.decode_csv(value, record_defaults=records)


    # 4、批处理，读取多条数据
    feature_batch, label_batch = tf.train.batch([data[0:-1], data[-1]], batch_size=10000, num_threads=1, capacity=10000)
    return feature_batch, label_batch


def ridge_regression(feature_batch, label_batch, lamda=0.5, learning_rate=0.01):
    '''
    Ridge Regression
    :param feature_batch: 特征
    :param label_batch: 标签
    :param lamda: 
    :param alpha: 
    :return: 
    '''
    weight = weight_init(shape=[feature_batch.shape[1], 1])
    bias = bias_init(shape=[1])
    y_hat = tf.matmul(feature_batch, weight) + bias
    loss = tf.reduce_mean(tf.square(label_batch-y_hat)) + lamda * tf.norm(weight)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(500):
            sess.run(train_op)
            if i % 100 == 0:
                print(f'第{i}次训练的权重为：{sess.run(weight)}，偏置为：{sess.run(bias)}')


def evaluate(weight, bias, feature_test, label_test):
    '''
    计算模型准确率
    :param weight: 权重
    :param bias: 偏置
    :param feature_test: 测试特征
    :param label_test: 测试标签
    :return: 准确率
    '''
    prediction = tf.matmul(feature_test, weight) + bias
    mse = tf.sqrt(tf.reduce_mean(tf.square(label_test-prediction)))

    return mse





features = ['car_brand', 'car_system', 'car_model', 'cylinder_number', 'driving', 'gearbox_type', 'intake_form', 'maximum_power',
 'voyage_range', 'car_class', 'vendor_guide_price', 'model_year', 'register_time', 'meter_mile', 'sell_times']




if __name__ == '__main__':
    # 1、构造文件列表
    file_name = os.listdir('/home/kdd/python/car')
    file_list = [os.path.join('/home/kdd/python/car', file) for file in file_name]
    # print(file_list)

    # 2、构造CSV阅读器读取CSV
    feature_batch, label_batch = read_csv(file_list)
    # print(feature_batch)

    # 3、开启会话
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取内容
        try:

            feature, label = sess.run([feature_batch, label_batch])
            ridge_regression(feature, label, lamda=0.5, learning_rate=0.0001)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        # 回收子线程
        coord.request_stop()
        coord.join(threads)