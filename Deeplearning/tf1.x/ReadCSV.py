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
	# reader = tf.data.TextLineDataset()
	key, value = reader.read(file_queue)

	# 3、对每行内容进行解码，record_default:指定每一个样本的每一列的类型
	records = [['None'],['None'],['None'],['None'],['None'],[0],[1.0],[0],[0]]
	sample = tf.decode_csv(records=value, record_defaults=records)
	
	# 4、批处理，读取多条数据
	sample_batch = tf.train.batch(sample, batch_size=100, num_threads=1, capacity=100)
	return sample_batch


if __name__ == '__main__':
	# 1、准备文件列表
	file_name = os.listdir('/home/kdd/python/data/csv')
	filelist = [os.path.join('/home/kdd/python/data/csv', file) for file in file_name]
	print(filelist)

	# 2、构造阅读器，批量读取csv
	sample_batch = csv_reader(filelist)

	# 3、开启会话运行
	with tf.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()
		
		# 开启读取文件的线程
		threads = tf.train.start_queue_runners(sess, coord=coord)

		# 打印读取的内容
		sample_batch = sess.run(sample_batch)
		for sample in sample_batch:
			sample_name = [f.decode('UTF-8') for f in sample]
			print(sample_name)

		# 回收线程
		coord.request_stop()
		coord.join(threads)


