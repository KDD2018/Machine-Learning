#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import reader
import time


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'PBTdata_dir',
    '/home/kdd/python/DATA/simple-examples/data',
    'PBT data path')


class PTBModel(object):
	def __init__(self, is_training, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)
		self.size = config.hidden_size
		self.vocab_size = config.vocab_size

		# 以LSTM结构作为循环体结构，并且在训练时使用dropout
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

		# 堆叠LSTM单元
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(config.num_layers)], state_is_tuple=True)

		# 初始化最初的状态，即全0的向量
		self.initial_state = cell.zero_state(batch_size, tf.float32)

		# 将单词ID转换为单词向量， 这里embedding为embedding_lookup()函数的维度信息
		# 单词总数通过vocab_size传入，每个单词向量的维度是size（参数hidden_size), 这样便得出embedding参数的维
		embedding = tf.get_variable('embedding', [self.vocab_size, self.size], dtype=tf.float32)

		# 通过embedding_lookup()函数将原本batch_size * num_steps个单词ID转换为单词向量，转换后的输入层的维度为batch_size * num_steps * size
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)

		# 在训练模式下，会对inputs添加dropout
		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		# 定义输出列表，在这里对不同时刻LSTM结构的输出进行汇总，之后通过一个全连接得到最终输出
		outputs = []
		# 定义state存储不同batch中LSTM的状态，并初始化为0
		state = self.initial_state
		 
		with tf.variable_scope('RNN'):
			for time_step in range(num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				# 从输入数据获取当前时刻的输入并传入LSTM结构
				cell_output, state = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		# concat()函数用于将输出的outputs展开成[batch_size, size * num_steps]的形状，然后再用reshape()转换为[batch_size * num_steps, size]的形状
		output = tf.reshape(tf.concat(outputs, 1), [-1, self.size])





train_data, valid_data, test_data, _ = reader.ptb_raw_data(FLAGS.PBTdata_dir)
result = reader.ptb_producer(train_data, 3, 4)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2):
        x, y = sess.run(result)
        print('input_data:\n', x)
        print('target:\n', y)
    coord.request_stop()
    coord.join(threads)
