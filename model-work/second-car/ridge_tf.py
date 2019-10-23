#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import time



def weight_init(shape):
    '''
    初始化权重
    :param shape: 权重的形状 
    :return: 初始化的权重值
    '''
    weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0), name='Weight')

    return weight


def bias_init(shape):
    '''
    初始化偏置
    :param shape: 偏置的形状 
    :return: 初始化的偏置值
    '''
    bias = tf.Variable(tf.constant(0.0, shape=shape), name='Bias')

    return bias


def read_csv(filelist):
    '''
    批获取csv数据
    :param filelist: 文件列表
    :return: 
    '''
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist, num_epochs=2)

    # 2、 构造CSV阅读器，读取队列数据
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
    # records = [['None'], ['None'], ['None'], [0], ['None'], ['None'], ['None'], [1.0], [1.0], ['None'], [1.0], [0],
    #            ['None'], [1.0], [1.0], [1.0]]
    records = [[0.0] for _ in range(679)]
    data = tf.decode_csv(value, record_defaults=records)

    # 4、批处理，读取多条数据
    feature_batch, label_batch = tf.train.batch([data[0:-1], data[-1]], batch_size=100, num_threads=1, capacity=1000)
    return feature_batch, label_batch


def ridge_regression(X):
    '''
    Ridge Regression
    :param X: 特征
    :return: y_hat 拟合值
    '''

    with tf.name_scope('Param_init'):
        weight = weight_init(shape=[678, 1])
        bias = bias_init(shape=[1])

    with tf.name_scope('Model'):
        y_hat = tf.matmul(X, weight) + bias

    # with tf.name_scope('MSE'):
    #     prediction = tf.matmul(x, weight) + bias
    #     mse = tf.sqrt(tf.reduce_mean(tf.square(y - prediction)))

    return y_hat, weight, bias


features = ['car_brand', 'car_system', 'car_model', 'cylinder_number', 'driving', 'gearbox_type', 'intake_form', 'maximum_power',
 'voyage_range', 'car_class', 'vendor_guide_price', 'model_year', 'register_time', 'meter_mile', 'sell_times']



if __name__ == '__main__':
    # 1、构造文件列表
    file_name = os.listdir('/home/kdd/python/car')
    file_list = [os.path.join('/home/kdd/python/car', file) for file in file_name]
    # print(file_list)

    # 2、构造CSV阅读器读取CSV
    feature_batch, label_batch = read_csv(file_list)
    # feature_test, label_test = read_csv([os.path.join('/home/kdd/python/car/test', 'suv_test.csv')])
    # print(feature_batch)


    # 3、构建Ridge模型
    y_hat, weight, bias = ridge_regression(feature_batch)

    # 4、计算损失, 优化
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.square(label_batch - y_hat)) + 0.5 * tf.norm(weight)

    with tf.name_scope('Optimizer'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=loss)

    # 4、计算模型精度
    # mse = evaluate(w, b, feature_test, label_test)

    # 初始化Variable
    init_op = tf.global_variables_initializer()

    # 、开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 循环训练
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_step = sess.run([train_op, loss])
                duration = time.time() - start_time
            if step % 500 == 0:
                print(f'\n第{step}次训练的损失为：{loss_step}({duration} sec)')
            step += 1
        except tf.errors.OutOfRangeError:
            print('Done reading -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
        # 回收子线程
        # coord.request_stop()
        # coord.join(threads)
