#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/kdd/python/MNIST_DATA', 'mnist data path')


mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)



def nn():
	# 1、准备数据
	with tf.variable_scope('Data'):
		x = tf.placeholder(tf.float32, [None, 784])
		y = tf.placeholder(tf.int32, [None, 10])

	# 2、初始化权重和偏置
	with tf.variable_scope('Initial'):
		w = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='Weight')
		b = tf.Variable(tf.constant(0.0, shape=[10]))
		y_hat = tf.matmul(x, w) + b

	# 3、求样本平均损失
	with tf.variable_scope('Loss'):
		loss_sum = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
		loss = tf.reduce_mean(loss_sum)

	# 4、梯度下降优化
	with tf.variable_scope('Optimizer'):
		train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	# 5、计算准确率
	with tf.variable_scope('Acc'):
		equal_list = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
		accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

	init_op = tf.global_variables_initializer()

	# 6、收集变量
	# scalar变量
	tf.summary.scalar('Cost', loss)
	tf.summary.scalar('Acc', accuracy)
	# 高维度变量
	tf.summary.histogram('Weights', w)
	tf.summary.histogram('Bias', b)

	# 定义合并变量的op
	merged = tf.summary.merge_all()



	# 开启会话训练
	with tf.Session() as sess:
		# 初始化变量
		sess.run(init_op)

		# 将变量写入events文件
		filewriter = tf.summary.FileWriter('/home/kdd/python/tmp', graph=sess.graph)

		# 迭代训练
		for i in range(2000):
			mnist_x, mnist_y = mnist.train.next_batch(100)
			sess.run(train_op, feed_dict={x: mnist_x, y: mnist_y})	

			# 将每次训练的参数写入events
			summary = sess.run(merged, feed_dict={x: mnist_x, y: mnist_y})
			filewriter.add_summary(summary, i)

			print('第%d步训练， 准确率为：%f'%(i, sess.run(accuracy, feed_dict={x: mnist_x, y:mnist_y})))

	return None


if __name__ == '__main__':
	nn()


