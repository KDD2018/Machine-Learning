#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf


# 1、先定义队列
Q = tf.FIFOQueue(3, tf.float32)

# 将数据放入队列
enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])

# 2、定义处理数据的逻辑: 取数据+1
out_q = Q.dequeue()  # 取出数据
data = out_q + 1  # 加一操作
en_q = Q.enqueue(data)  # 放入队列


# 开启会话
with tf.Session() as sess:
	# 初始化队列
	sess.run(enq_many)
	
	# 处理数据+1
	for i in range(100):
		sess.run(en_q)

	# 训练数据
	for i in range(Q.size().eval()):
		print(sess.run(Q.dequeue()))

