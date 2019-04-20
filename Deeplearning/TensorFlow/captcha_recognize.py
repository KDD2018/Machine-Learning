#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'captcha_dir',
    '/home/kdd/python/DATA/tfrecords/captcha.tfrecords',
    'CAPTCHA data of tfrecords ')
tf.app.flags.DEFINE_integer('batch_size', 100, "samples'number of every batch")
tf.app.flags.DEFINE_integer('letter_num', 26, "possibility of every target")
tf.app.flags.DEFINE_integer('label_num', 4, "number of every label")


def weight_init(shape):
    '''
    权重初始化API
    '''
    weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return weight


def bias_init(shape):
    '''
    偏置初始化API
    '''
    bias = tf.Variable(tf.constant(0.0, shape=shape))
    return bias


def captcha_read():
    '''
    读取验证码数据API
    returns: image_batch, label_batch
    '''
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer([FLAGS.captcha_dir])

    # 2、构造tfrecords文件阅读器
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)

    # 3、解析example协议块
    features_dict = {'image': tf.FixedLenFeature([], tf.string),
                     'label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(value, features=features_dict)

    # 4、解码tfrecords文件
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    # 固定图片形状
    image_shape = tf.reshape(image, [20, 80, 3])
    label_shape = tf.reshape(label, [4])

    # 5、批处理
    image_batch, label_batch = tf.train.batch(
        [iamge_shape, label_shape], batch_size=FLAGS.batch_size, num_threads=1, capacity=FLAGS.batch_size)
    print(image_batch, label_batch)

    return image_batch, label_batch


def fc_model(image):
    '''
    构建全连接神经网络
    returns: y_hat [100, 4*26]
    '''
    with tf.variable_scope('model'):
        # 1、随机初始化权重偏置
        weights = weight_init([20 * 80 * 3, 4 * 26])
        bias = bias_init([4 * 26])

        # 2、转换验证码数据shape为[-1, 20*80*3], dtype为float32
        image_reshape = tf.cast(
            tf.reshpe(image, [-1, 20 * 80 * 3]), tf.float32)

        # 3、全连接
        y_hat = tf.matmul(image_reshape, weights) + bias

    return y_hat


def captchaRec():
    '''
    全连接神经网络识别验证码
    '''
    # 1、读取验证码数据
    image_batch, label_batch = captcha_read()

    # 2、全连接神经网络预测
    y_hat = fc_model(image_batch)

    # 3、目标值one-hot编码转换[-1, 4, 26]
    y_true = tf.one_hot(
        label_batch,
        depth=FLAGS.letter_num,
        axis=2,
        on_value=1.0)

    # 4、计算交叉熵损失
    with tf.variabel_scope('loss'):
        cross_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(
            y_true, [FLAGS.batch_size, FLAGS.label_number * FLAGS.letter_number]))
        loss = tf.reduce_mean(cross_loss)

    # 5、梯度下降，优化损失
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 6、计算准确率
    equal_list = tf.equal(
        tf.argmax(
            y_true, 2), tf.argmax(
            tf.reshape(
                y_hat, [
                    FLAGS.batch_size, FLAGS.label_num, FLAGS.letter_num]), 2))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 7、开启会话
    with tf.Session() as sess:
        sess.run(init_op)
        # 定义线程协调器
        coord = tf.train.Coordinator()
        # 开启线程读取文件
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 循环训练
        for i in range(5000):
            sess.run(train_op)
            print('第%d次训练，准确率为: %f' % (i, sess.run(accuracy)))

        # 回收线程
        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == '__main__':
    captchaRec()
