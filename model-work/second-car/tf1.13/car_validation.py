#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import time
# from ridge_tf import Model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_data_dir', '/home/kdd/python/car/test', 'path for test_data')
# tf.app.flags.DEFINE_integer('batch_size', 200, 'number of samples for validation')


def get_file_list(path):
    '''
    返回文件列表
    :param path: 文件所在路径
    :return: 文件列表
    '''
    # 构造文件列表
    file_name = os.listdir(path)
    file_list = [os.path.join(path, file) for file in file_name]
    # print(file_list)
    return file_list


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
    bias = tf.Variable(tf.constant(0.0, shape=shape), name='bias')

    return bias


def read_csv(path):
    '''
    批获取csv数据
    :param filelist: 文件列表 
    :return: 批量特征  批量标签
    '''
    # 1、构造文件列表
    file_list = get_file_list(path)

    # 2、构造文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 3、 构造CSV阅读器，读取队列数据
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 4、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
    records = [[0.0] for _ in range(698)]
    data = tf.decode_csv(value, record_defaults=records)

    # 5、批处理，读取多条数据
    feature_batch, label_batch = tf.train.shuffle_batch([data[0:-1], data[-1]], batch_size=200,
                                                        num_threads=1, capacity=1000,
                                                        min_after_dequeue=3)
    # print(feature_batch, label_batch)
    return feature_batch, label_batch


def predict(X, weight, bias):
    '''
    预测
    :param X: 待测样本特征
    :param weight: 学得权重
    :param bias: 学得偏置
    :return: 预测值
    '''
    y_hat = tf.matmul(X, weight) + bias

    return y_hat


def r2_score(label, y_hat):
    '''
    计算拟合优度
    :param label: 样本真实标签
    :param y_hat: 预测值
    :return: 拟合优度
    '''

    sse = tf.reduce_sum(tf.square(label - y_hat))
    sst = tf.reduce_sum(tf.square(label - tf.reduce_mean(label)))
    R2 = tf.constant(1.0) - sse / sst

    return R2


def load_model(model_path):
    '''
    加载模型
    :param model_path: 模型路径
    :return: 权重、偏置
    '''
    ckpt = tf.train.get_checkpoint_state('../model-param/')

    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)

    graph = tf.get_default_graph()

    w = sess.run(graph.get_tensor_by_name('Model/weight:0'))
    b = sess.run(graph.get_tensor_by_name('Model/bias:0'))

    return w, b


if __name__ == '__main__':

    # 读取
    feature_test_batch, label_test_batch = read_csv(FLAGS.test_data_dir)

    # w = tf.placeholder(dtype=tf.float32, shape=[697,1], name='weight')
    # b = tf.placeholder(dtype=tf.float32, shape=[1], name='bias')

    # prediction = predict(feature_test_batch, w, b)
    #
    # R2 = r2_score(label_test_batch, prediction)


    var_init = tf.global_variables_initializer()

    # reader = tf.train.NewCheckpointReader(ckpt_path)
    # all_variables = reader.get_variable_to_shape_map()


    with tf.Session() as sess:

        sess.run(var_init)
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state('../model-param/')

        # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        # saver.restore(sess, ckpt.model_checkpoint_path)
        #
        # graph = tf.get_default_graph()
        #
        # w_ = sess.run(graph.get_tensor_by_name('Model/weight:0'))
        # b_ = sess.run(graph.get_tensor_by_name('Model/bias:0'))
        # # print(w)
        # print(b)


        reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
        all_vars = reader.get_variable_to_shape_map()
        reader.get
        print(all_vars)


        # # 定义一个线程协调器
        # coord = tf.train.Coordinator()
        # # 开启读取文件的线程
        # threads = tf.train.start_queue_runners(sess, coord=coord)
        #
        # # 循环训练
        # try:
        #     num = 0
        #     while not coord.should_stop():
        #         goodness_fit = sess.run([R2], feed_dict={w: w_, b: b_})
        #         print(f'测试集上的拟合优度{goodness_fit}')
        #         num += 1
        # except tf.errors.OutOfRangeError:
        #     print(f'\nDone reading for {num} example.')
        # finally:
        #     coord.request_stop()
        #
        # coord.join(threads)

