#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

def csv_reader(filelist):
	'''
	读取CSV文件
	filelist: 文件路径+名字的列表
	return: 数据内容
	'''
	# 1、构造文件队列
	file_queue = tf.train.string_input_producer(filelist)
	# file_queue = tf.data.Dataset.from_tensor_slices(filelist)
	# 2、 构造CSV阅读器，读取队列数据
	reader = tf.TextLineReader()
	key, value = reader.read(file_queue)

	# value = tf.data.TextLineDataset(file_queue)
	# print(value)

	# 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
	records = [['None'],['None'],['None'],['None'],['None'],['None'],['None'],['None'],['None']]
	clk = tf.decode_csv(value, record_defaults=records, use_quote_delim=False)
	
	# 4、批处理，读取多条数据
	clk = tf.train.batch(clk, batch_size = 10, num_threads=1, capacity=100)
	clk = tf.strings.unicode_decode(clk, input_encoding='UTF-8')
	return clk

if __name__ == '__main__':
	# 1、准备文件列表
	file_name = os.listdir('/home/kdd/data')
	filelist = [os.path.join('/home/kdd/data', file) for file in file_name]
	# print(filelist)
	clk = csv_reader(filelist)

	# 开启会话运行
	with tf.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()
		
		# 开启读取文件的线程
		threads = tf.train.start_queue_runners(sess, coord=coord)

		# 打印读取的内容
		print(sess.run(clk))

		# 回收线程
		coord.request_stop()
		coord.join(threads)


