#!/usr/bin/python3
# -*- coding: utf-8 -*-



import tensorflow as tf
import os

def myregress():
    '''
    运用Tensorflow实现简单现行回归
    :return: None
    '''

    with tf.variable_scope('data'):
        # 1、 准备特征值和目标值
        X = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='Feature')
        y = tf.matmul(X, [[0.7]]) + 0.8

    with tf.variable_scope('model'):
        # 2、 建立模型
        # 权重、偏置随机初始化
        weight = tf.Variable(tf.random_normal([1,1], mean=.0, stddev=1.0), name='Weight', trainable=True)
        bias = tf.Variable(0.0, name='bias', trainable=True)
        # 预测值
        y_hat = tf.matmul(X, weight) +bias

    with tf.variable_scope('loss'):
        # 3、 求损失函数
        loss = tf.reduce_mean(tf.square(y_hat-y))

    with tf.variable_scope('optimizer'):
        # 4、 根据梯度下降法优化损失函数
        train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 5、 变量初始化(变量一定要初始化)
    init_op = tf.global_variables_initializer()

    # 收集显示变量的信息
    tf.summary.histogram('Weights', weight)
    tf.summary.scalar('Loss', loss)

    # 合并变量
    merged = tf.summary.merge_all()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()


    # 6、 通过会话运行程序
    with tf.Session() as sess:
        # 运行初始化变量
        sess.run(init_op)
        # 打印权重、偏置的随机初始化值
        print('权重初始化值： %f, 偏置初始化值： %f'%(sess.run(weight), sess.run(bias)))
        # 建立事件文件
        filewriter = tf.summary.FileWriter('./tmp/summary', graph=sess.graph)

        # 加载模型
        if os.path.exists('./tmp/tf_model/checkpoint'):
            saver.restore(sess, './tmp/tf_model/linearmodel')
        # 循环训练 运行优化op
        for i in range(200):
            sess.run(train_op)
            # 运行合并变量
            summ = sess.run(merged)
            filewriter.add_summary(summ, i)
            print('第%d次参数权重： %f, 参数偏置： %f'%(i, sess.run(weight), sess.run(bias)))
        saver.save(sess, './tmp/tf_model/linearmodel')

    return None


if __name__ == '__main__':
    myregress()