#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('captcha_dir', '/home/kdd/python/CAPTCHA', 'CAPTCHA data path')
tf.app.flags.DEFINE_string('letter', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '验证码字符集')
tf.app.flags.DEFINE_string('tfrecords_dir', '/home/kdd/python/tfrecords/captcha.tfrecords', 'CAPTCHA tfrecords file')


def get_captcha_image():
	'''
	获取验证码图片数据
	Args: file_list 路径+文件名列表
	Return: image_batch 批量验证码图片
	'''
	# 构造文件名
	filename = [str(i)+'.jpg' for i in range(6000)]

	# 构造路径文件列表
	file_list = [os.path.join(FLAGS.captcha_dir, file) for file in filename]

	# 构造文件队列
	file_queue = tf.train.string_input_producer(file_list, shuffle=False)

	# 构造图片阅读器
	reader = tf.WholeFileReader()

	# 读取图片内容
	key, value = reader.read(file_queue)

	# 对图片内容进行解码
	image = tf.image.decode_jpeg(value)
	image.set_shape([20, 80, 3])

	# 批处理
	image_batch = tf.train.batch([image], batch_size=6000, num_threads=1, capacity=6000)
	
	return image_batch
	

def get_captcha_label():
	'''
	读取验证码图片的标签数据
	Args:
	Returns：labels 批量验证码标签
	'''
	# 构造文件队列
	file_queue = tf.train.string_input_producer(['/home/kdd/python/CAPTCHA'], shuffle=False)	
	# 构造CSV文件阅读器
	reader = tf.TextLineReader()
	# 读取CSV文件
	key, value = reader.read(file_queue)
	# 解码CSV文件
	number, label = tf.decode_csv(value, record_defaults=[[1], ['None']])
	# 批处理
	labels = tf.train.batch([label], batch_size=6000, num_threads=1, capacity=6000)

	return labels


def dealwithlabel(label_str):
	'''
	将字符串标签转换为int数字标签
	Args: label_str 字符串标签
	Returns: label_batch int数字标签
	'''
	# 构建字符索引{0: 'A', 1: 'B', ...}
	num_letter = dict(enumerate(list(FLAGS.letter)))
	# 键值对反转
	letter_num = dict(zip(num_letter.values(),num_letter.keys()))

	# 构建标签列表
	labels = []
	# 将字符串转为int
	for string in label_str:
		letter_list = [letter_num[letter] for letter in string.decode('utf-8')]
		array.append(letter_list)
	print(labels)
	# 将列表转换为Tensor
	label_batch = tf.constant(labels)

	return label_batch
			

def write2tfrecords(image_batch, label_batch):
	'''
	将验证码图片内容和标签写入tfrecords
	Args: image_batch 验证码图片的特征值
		  label_batch 验证码标签
	Returns: None
	'''
	# 将验证码标签转为uint8
	label_batch = tf.cast(label_batch, tf.uint8)
	print(label_batch)

	# 构造TFRecords存储器
	writer = tf.python.io.TFRecordsWriter(FLAGS.tfrecords_dir)

	# 循环构造每张验证码图片数据的example协议块,并序列化
	for i in range(6000):
		# 将每张图片数据的特征值转换成字符串形式
		image_string = image_batch[i].eval().tostring()
		# 标签值转换为字符串
		label_string = label_batch[i].eval().tostring()
		
		# 构造example协议块
		feature = {
					'image': tf.train.Feature(bytes_list=tf.train.Byteslist(value=[image_string]))
					'label': tf.train.Feature(bytes_list=tf.train.Byteslist(value=[label_string]))}
		example = tf.train.Example(features=tf.train.Features(feature=feature))
		
		writer.write(example.SerializeToString())
	# 关闭文件
	writer.close()

	return None


if __name__ = '__main__':
	# 获取验证码图片
	image_batch = get_captcha_image()

	# 获取验证码标签数据
	labels = get_captcha_label()

	with tf.Session() as sess:
		# 开启线程协调器
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		# 将字符串label处理成int数字
		label_str = sess.run(labels)
		print(label_str)
		label_batch = dealwithlabel(label_str)
		print(label_batch)

		# 将图片数据和标签内容写入tfrecords文件
		write2tfrecordds(image_batch, label_batch)
		
		# 关闭子线程
		coord.request_stop()
		coord.join(threads)

