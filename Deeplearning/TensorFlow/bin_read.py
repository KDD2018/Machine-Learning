#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os


FLAGS = tf.app.flags.FLAGS # 用于命令行传递参数
tf.app.flags.DEFINE_string('BinData_dir', '/home/kdd/python/cifar-10-batches-bin', '二进制数据文件路径')
tf.app.flags.DEFINE_string('TFRecord', '/home/kdd/python/tfrecords/cifar.tfrecords', '写入tfrecords文件的路径')


class BinReadTFrecords():
	'''
	读取二进制数据转换成tfrecords文件并读取tfrecords
	'''
	def __init__(self, filelist):
		# 文件列表
		self.file_list = filelist
		# 设定读取的图片的属性
		self.height = 32
		self.width = 32
		self.channel = 3
		self.label_bytes = 1
		self.image_bytes = self.height * self.width * self.channel
		self.bytes = self.image_bytes + self.label_bytes
	
	def read_bin_decode(self):
		'''
		读取二进制文件并解码
		'''
		# 1、构造文件队列
		file_queue = tf.train.string_input_producer(self.file_list)

		# 2、构造二进制文件阅读器
		reader = tf.FixedLengthRecordReader(record_bytes=self.bytes)
		key, value = reader.read(queue=file_queue)
		print(value)
	
		# 3、对图片的二进制数据进行解码
		image_bin = tf.decode_raw(bytes=value, out_type=tf.uint8)
		print(image_bin)

		# 4、将数据分为特征值和目标值
		label = tf.cast(tf.slice(image_bin, [0], [self.label_bytes]), tf.int32)
		image = tf.slice(image_bin, [self.label_bytes], [self.image_bytes])

		# 5、设定图片样本特征数据的形状
		image_shape = tf.reshape(image, [self.height, self.width, self.channel])
		print(image_shape)

		# 6、批处理
		image_batch, label_batch = tf.train.batch([image_shape, label], batch_size=100, num_threads=1, capacity=100)
		print(image_batch, label_batch)
		return (label_batch, image_batch)

	def write2tfrecords(self, image_batch, label_batch):
		'''
		将图片的特征值和目标值写入tfrecords
		Args: image_batch 100张图片的特征值
			  label_batch 100张图片的目标值
		'''
		# 1、建立TFRecord存储器
		writer = tf.python_io.TFRecordWriter(FLAGS.TFRecord)

		# 2、构造example协议,循环写入tfreords
		for i in range(100):
			# 取出第i张图片的特征值和目标值
			image = image_batch[i].eval().tostring()
			label = label_batch[i].eval()[0]
			
			# 构造一个样本的example协议
			feature_dict = {
				'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
				'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
			example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

			# 将单个样本写入文件
			writer.write(example.SerializeToString())
		
		# 关闭
		writer.close()
		return None

	def read_tfrecords(self):
		'''
		读取tfrecords文件
		'''
		# 1、构造文件队列
		file_queue = tf.train.string_input_producer([FLAGS.TFRecord])

		# 2、构造tfrecords文件阅读器
		reader = tf.TFRecordReader()
		key, value = reader.read(file_queue)

		# 3、解析example
		features_dict = {
			'image':tf.FixedLenFeature([], tf.string),
			'label':tf.FixedLenFeature([], tf.int64)}
		features = tf.parse_single_example(value, features=features_dict)

		# 4、解码内容，读取的内容格式是string需要解码，int64，float32不需要解码
		image = tf.decode_raw(features['image'], tf.uint8)
		print(image)
		# 固定图片形状
		image_shape = tf.reshape(image, [self.height, self.width, self.channel])
		label = features['label']
		print(image_shape, label)
		
		# 5、批处理
		image_batch, label_batch = tf.train.batch([image_shape, label], batch_size=100, num_threads=1, capacity=100)
		
		return image_batch, label_batch


if __name__ == '__main__':

	# 1、准备文件列表
	file_name = os.listdir(FLAGS.BinData_dir)
	filelist = [os.path.join(FLAGS.BinData_dir, file) for file in file_name if file[-3:]=='bin']
	
	# 2、读取二进制数据
	br = BinReadTFrecords(filelist)
	# label_batch, image_batch = br.read_bin_decode()
	
	# 2、读取tfrecords文件
	image_batch, label_batch = br.read_tfrecords()

	# 3、开启会话，执行读取过程
	with tf.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()
		
		# 开启读取文件的线程
		threads = tf.train.start_queue_runners(sess, coord=coord)

		# 存入tfrecords文件
		# print('开始写入')
		# br.write2tfrecords(image_batch, label_batch)
		# print('写入完成')
		
		# 打印读取的图片数据
		print(sess.run([image_batch, label_batch]))

		# 回收子线程
		coord.request_stop()
		coord.join(threads)
