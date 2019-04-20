#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf


# 模拟异步 子线程存数据，主线程读数据

# 1、定义队列，1000个数据
Q = tf.FIFOQueue(1000, tf.float32)

# 2、定义要做的事情：循环加一，存入队列
tmp = tf.Variable(0.0)
data = tf.assign_add(tmp, tf.constant(1.0))
en_q = Q.enqueue(data)

# 3、定义队列管理器，指定子线程的任务
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q]*2)

# 初始化变量
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init_op)

	# 开启线程管理器
	coord = tf.train.Coordinator()	
	# 开启子线程
	threads = qr.create_threads(sess, coord=coord, start=True)
	
	# 读取数据
	for i in range(300):
		print(sess.run(Q.dequeue()))
	
	# 回收资源
	coord.request_stop()
	coord.join(threads)
