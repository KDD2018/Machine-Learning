#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os


def pictureRead(filelist):
	'''
	读取图片转换成张量
	Args: filelist 文件路径+文件名的列表
	Returns: 图片内容张量
	'''
	# 1、构造文件队列
	file_queue = tf.train.string_input_producer(filelist)
	
	# 2、构造阅读器
	reader = tf.WholeFileReader()
	key, value = reader.read(file_queue)

	# 3、对图片数据进行解码
	image = tf.image.decode_jpeg(value)
	print(image)

	# 4、统一图片大小
	image_size = tf.image.resize_images(image, [200, 200])
	# 设定图片样本数据的形状
	image_size.set_shape([200, 200, 3])
	print(image_size)

	# 5、批处理
	image_batch = tf.train.batch([image_size], batch_size=100, num_threads=1, capacity=100)
	print(image_batch)
	return (image, image_batch)


if __name__ == '__main__':

	# 1、准备文件列表
	file_name = os.listdir('/home/kdd/python/beauty')
	filelist = [os.path.join('/home/kdd/python/beauty', file) for file in file_name]
	image, image_batch = pictureRead(filelist)
	
	# 2、开启会话，读取图片文件
	with tf.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()
		
		# 开启读取文件的线程
		threads = tf.train.start_queue_runners(sess, coord=coord)

		# 打印读取的图片数据
		print(sess.run([image_batch]))

		# 回收子线程
		coord.request_stop()
		coord.join(threads)
