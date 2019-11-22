#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import time
from generate_csv import GenerateCSV



def get_file_list(path):
    '''
    返回文件列表
    :param path: 文件所在路径
    :return: 文件列表
    '''
    file_name = os.listdir(path)
    file_list = [os.path.join(path, file) for file in file_name]

    return file_list


def weight_init(shape):
    '''
    初始化权重
    :param shape: 权重的形状 
    :return: 初始化的权重值
    '''
    return tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0), dtype=tf.float32, name='weight')


def bias_init(shape):
    '''
    初始化偏置
    :param shape: 偏置的形状 
    :return: 初始化的偏置值
    '''
    return tf.Variable(tf.constant(0.0, shape=shape), dtype=tf.float32, name='bias')


def read_csv(path, num_cols, batch_size=10, threads=1, capacity=500, min_after_dequeue=200):
    '''
    批获取csv数据
    :param filelist: 文件列表 
    :return: 批量特征  批量标签
    '''
    # 1、构造文件列表
    file_list = get_file_list(path)

    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(file_list, num_epochs=10)

    # 2、 构造CSV阅读器，读取队列数据
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
    records = [[0.0] for _ in range(num_cols)]
    data = tf.decode_csv(value, record_defaults=records)

    # 4、批处理，读取多条数据
    feature_batch, label_batch = tf.train.shuffle_batch([data[0:-1], data[-1]], batch_size=batch_size, num_threads=threads,
                                                        capacity=capacity, min_after_dequeue=min_after_dequeue)
    # print(feature_batch.shape, label_batch.shape)
    return feature_batch, label_batch


def ridge_regression(X, num_cols):
    '''
    Ridge Regression
    :param X: 特征
    :return: y_hat 拟合值
    '''

    weight = weight_init(shape=[num_cols-1, 1])
    bias = bias_init(shape=[1, 1])
    y_hat = tf.squeeze(tf.matmul(X, weight) + bias, axis=1, name='y_hat')

    return y_hat, weight, bias


def r2_score(label, y_hat):
    '''
    计算拟合优度
    :param label: 样本真实标签
    :param y_hat: 预测值
    :return: 拟合优度
    '''
    label = tf.cast(label, dtype=tf.float32)
    y_hat = tf.cast(y_hat, dtype=tf.float32)
    sse = tf.reduce_sum(tf.square(label - y_hat))
    sst = tf.reduce_sum(tf.square(label - tf.reduce_mean(label)))
    R2 = 1.0 - sse / sst

    return R2


def train_and_save_model(train_data_path, columns, model_path, alpha=0.01, learning_rate=0.01):
    '''
    训练模型并持久化
    :param path: 数据源路径
    :return: None
    '''
    with tf.name_scope('Input_data'):
        # 构造CSV阅读器读取CSV
        feature_batch, label_batch = read_csv(train_data_path, num_cols=columns)
        # feature_test, label_test = read_csv(test_data_path,  num_cols=columns, batch_size=batch_size)

    with tf.name_scope('Model'):
        # 构建Ridge模型
        y_hat, weight, bias = ridge_regression(feature_batch, num_cols=columns)

    with tf.name_scope('Loss'):
        # 计算并优化损失
        cost = tf.reduce_mean(tf.square(label_batch - y_hat))
        regular_l2 = tf.norm(weight)
        loss = tf.add(cost, alpha * regular_l2, name='loss')

    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

    # with tf.name_scope('R2_score'):
    #     R2 = r2_score(label_test, tf.squeeze(tf.matmul(feature_test, weight) + bias, axis=1))

    with tf.name_scope('Log_summary'):
        # 日志摘要
        tf.summary.scalar('Cost', loss)  # scalar单个标量值
        tf.summary.histogram('Weights', weight)  # histogram高维度变量
        tf.summary.histogram('Bias', bias)

        # 合并默认图中收集的所有摘要
        merged = tf.summary.merge_all()

    # 初始化Variable
    init_op = tf.global_variables_initializer()

    with tf.name_scope('Save'):
        # 实例化Saver类
        saver = tf.train.Saver()
        tf.add_to_collection('prediction', y_hat)

    # 开启会话执行
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())

        # 定义FileWriter， 用于将协议缓冲区摘要写入events文件
        file_writer = tf.summary.FileWriter('/home/kdd/python/tmp', graph=sess.graph)

        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 循环训练
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                train, loss_step, summary = sess.run([train_op, loss, merged])

                # 将每次训练产生的协议缓冲区摘要写入events文件
                summary = sess.run(merged)
                file_writer.add_summary(summary, step)
                if step % 200 == 0:
                    duration = time.time() - start_time
                    print(f'\n第{step}次训练的损失为：{loss_step:.4f}, 耗时{duration:.3f} sec')
                # if step * batch_size % len_train_data == 0:
                #     epoch = step * batch_size / len_train_data
                #     print(f'Epoch: {epoch}, 拟合优度为{sess.run(R2)}')
                    saver.save(sess, model_path, global_step=step)
                step += 1
            # saver.save(sess, FLAGS.model_dir, global_step=step)
        except tf.errors.OutOfRangeError:
            print(f'\nDone training for {step} steps.')
            # print(e)
        finally:
            coord.request_stop()
        file_writer.close()
        coord.join(threads)


def run():
    # 1、读取MySQL并清洗然后写入CSV
    generateCSV = GenerateCSV()
    num_cols, car_class, len_train_data = generateCSV.run()

    if num_cols:
        # 数据路径
        train_data_dir = f'/home/kdd/python/car/{car_class}'
        test_data_dir = f'/home/kdd/python/car/{car_class}_test/'
        # 模型路径
        model_saving_dir = f'../../model-param/{car_class}/{car_class}.ckpt'

        # 　２、读取CSV并训练模型
        train_and_save_model(train_data_path=train_data_dir, columns=num_cols[0], model_path=model_saving_dir)



if __name__ == '__main__':

    run()