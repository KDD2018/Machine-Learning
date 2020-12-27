#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
卷积层一：
		卷积: 32个5*5的Filter， stride 1, padding='SAME'    输入：[None, 28, 28, 1]		输出: [None, 28, 28, 32]    bias=32
		激活： [None, 28, 28, 32]
		池化: 2*2, stride 2, padding='SAME'	[None, 28, 28, 32]---------->[None, 14, 14, 32]
卷积层二：
		卷积：64个5*5的Filter， stride 1, padding='SAME'		输入： [None, 14, 14, 32]   输出： [None, 14, 14, 64]  bias=64
		激活：[None, 14, 14, 64]
		池化：2*2, stride 2  padding='SAME'  [None, 14, 14, 64]----------->[None, 7, 7, 64]
全连接层：
		[None, 7*7*64]  [7*7*64, 10]  [None, 10]   bias=10
'''
 
 
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/kdd/python/DATA/MNIST_DATA', 'mnist data path') 


def weight_init(shape):
	'''
	初始化权重的函数
	'''
	weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
	return weight

def bias_init(shape):
	'''
	初始化偏置的函数
	'''
	bias = tf.Variable(tf.constant(0.0, shape=shape))
	return bias


def cnn_model():
	'''
	自定义卷积模型
	'''
	# 1、准备数据占位符 x [None, 784] y [None, 10]
	with tf.variable_scope('data'):
		x = tf.placeholder(tf.float32, [None, 784])
		y = tf.placeholder(tf.int32, [None, 10])

	# 2、卷积层一  卷积、激活、池化
	with tf.variable_scope('conv1'):
		# 随机初始化权重、偏置
		w_conv1 = weight_init([5, 5, 1, 32])
		b_conv1 = bias_init([32])		
		print(w_conv1, b_conv1)
		# 重置特征值的形状 [None, 784]---->[None, 28,28,1]
		x_reshape = tf.reshape(x, [-1, 28, 28, 1])
		# 卷积 [None, 28, 28, 1]----->[None, 28, 28, 32]
		conv1 = tf.nn.conv2d(input = x_reshape, filter=w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
		# 激活
		activ1 = tf.nn.relu(conv1)
		# 池化 [None, 28, 28, 32]------>[None, 14, 14, 32]
		pool1 = tf.nn.max_pool(value=activ1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(pool1)

	# 3、卷积层二	
	with tf.variable_scope('conv2'):
		
		# 随机初始化权重、偏置
		w_conv2 = weight_init([5, 5, 32, 64])
		b_conv2 = bias_init([64])		
		# 卷积 [None, 14, 14, 32]------>[None, 14, 14, 64]
		conv2 = tf.nn.conv2d(input=pool1, filter=w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
		# 激活
		activ2 = tf.nn.relu(conv2)
		# 池化 [None, 14, 14, 64]----->[None, 7, 7, 64]
		pool2 = tf.nn.max_pool(value=activ2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	# 4、全连接层
	with tf.variable_scope('fc'):
		# 初始化权重、偏置
		w_fc = weight_init([7*7*64, 10])
		b_fc = bias_init([10])
		# 重置特征值形状 [None, 7, 7, 64]----->[None, 7*7*64]
		x_fc_reshape = tf.reshape(pool2, [-1, 7*7*64])

		# 全连接
		y_hat = tf.matmul(x_fc_reshape, w_fc) + b_fc
	
	return x, y, y_hat


def conv_fc():
	'''
	构建卷积神经网络
	'''
	# 建立卷积模型
	x, y, y_hat = cnn_model()
	
	# 计算交叉熵损失 
	with tf.variable_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

	# 梯度下降，优化损失
	with tf.variable_scope('optimizer'):
		train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

	# 计算准确率
	with tf.variable_scope('acc'):
		equal_list = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
		accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

	init_op = tf.global_variables_initializer()
	# 开启会话
	with tf.Session() as sess:
		sess.run(init_op)
		# 循环训练
		for i in range(2000):
			mnist_x, mnist_y = mnist.train.next_batch(100)
			sess.run(train_op, feed_dict={x: mnist_x, y: mnist_y})
			print('第%d次训练，准确率为：%f'%(i, sess.run(accuracy, feed_dict={x: mnist_x, y: mnist_y})))

	return None


if __name__ == '__main__':
	# 获取手写数字识别数据集MNIST
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	# 卷积神经网络
	conv_fc()


