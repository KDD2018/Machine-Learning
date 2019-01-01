#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt


def sig(x):
	'''
	Desc: sigmoid函数
	Args:
	    x(mat/float) 自变量
	Returns:
	    sigmoid sigmoid函数值
	'''

	sigmoid = 1.0 / (1 + np.exp(-x))
	return sigmoid

def partial_sig(x):
	'''
	Desc: Sigmoid导函数的值
	Args:
	    x(mat/float) 自变量
	Returns:
	    out Sigmoid导函数的值 
	'''

	m, n = np.shape(x)
	out = np.mat(np.zeros((m, n)))
	for i in range(m):
		for j in range(n):
			out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))
	return out

def hidden_in(feature, w0, b0):
	'''
	Desc: 计算隐含层的输入
	Args:
	    feature(mat) 特征
	    w0(mat) 输入层到隐含层之间的权重
	    b0(mat) 输入层到隐含层之间的偏置
	Returns: hidden_in 隐含层的输入
	'''

	m = np.shape(feature)[0]
	hidden_in = feature * w0
	for i in range(m):
		hidden_in[i, ] += b0
	return hidden_in

def hidden_out(hidden_in):
	'''
	Desc: 隐含层的输出
	Args: 
	    hidden_in(mat) 隐含层的输入
	Returns: 隐含层的输出
	'''

	hidden_out = sig(hidden_in)
	return hidden_out

def predict_in(hidden_out, w1, b1):
	'''
	Desc: 计算输出层的输入
	Args:
	    hidden_out(mat) 隐含层的输出
	    w1(mat) 隐含层到输出层之间的权重
	    b1(mat) 隐含层到输出层之间的偏置
	Returns:
	    predict_in(mat) 输出层的输入
	'''

	m = np.shape(hidden_out)[0]
	predict_in = hidden_out * w1
	for i in range(m):
		predict_in[i, ] += b1
	return predict_in

def predict_out(predict_in):
	'''
	Desc: 计算输出层的输出
	Args:
	    predict_in(mat) 输出层的输入
	Returns:
	    result 输出层的输出
	'''

	result = sig(predict_in)
	return result

def get_cost(cost):
	'''
	Desc: 计算当前损失函数的值
	Args:
	    cost(mat) 预测值与标签值之间的差
	Returns:
	    cost_avg 损失函数的值
	'''

	cost_sum = 0.0
	m, n = np.shape(cost)
	for i in range(m):
		for i in range(n):
			cost_sum += cost[i, j] * cost[i, j]
	cost_avg = cost_sum / m
	return cost_avg

def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    '''
 	Desc: 计算隐含层的输入
 	Args: 
 	     feature(mat) 特征
 	     label(mat) 标签
 	     n_hidden(int) 隐含层的节点个数
 	     maxCycle(int) 最大的迭代次数
 	     alpha(float) 学习率
 	     n_output(int) 输出层的节点个数
    Returns:
        w0(mat) 输入层到隐含层之间的权重
        b0(mat) 输入层到隐含层之间的偏置
        w1(mat) 隐含层到输出层之间的权重
        b1(mat) 隐含层到输出层之间的偏置
 	'''

    m, n = np.shape(feature)
 	# 1 初始化
    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - np.mat(np.ones((n, n_hidden))) *\
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - np.mat(np.ones((1, n_hidden))) *\
         (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - np.mat(np.ones((n_hidden, n_output))) *\
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - np.mat(np.ones((1, n_output))) *\
         (4.0 * sqrt(6) / sqrt(n_hidden + n_output))

	# 2 训练
    i = 0
    while i <= maxCycle:    	
        # 2.1 信号正向传播
        # 2.1.1 计算隐含层的输入
        hidden_input = hidden_in(feature, w0, b0)
        # 2.1.2 计算隐含层的输出
        hidden_output = hidden_out(hidden_input)
        # 2.1.3 计算输出层的输入
        output_in = predict_in(hidden_output, w1, b1)
        # 2.1.4 计算输出层的输出
        output_out = predict_out(output_in)

		# 2.2 误差的反向传播
		# 2.2.1 隐含层到输出层之间的残差
        delta_output = -np.multiply((label - output_out), partial_sig(output_in))
		# 2.2.2 输入层到隐含层之间的残差
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(hidden_input))

		# 2.3 修正权重和偏置
        w1 = w1 - alpha * (hidden_output.T * delta_output)
        b1 = b1 - alpha * np.sum(delta_output, axis=0) * (1.0 / m)
        w0 = w0 - alpha * (feature.T * delta_hidden)
        b0 = b0 - alpha * np.sum(delta_hidden, axis=0) * (1.0 / m)
        if i % 100 == 0:
        	print('\t---------iter:', i, ', cost: ', (1.0 / 2) *\
        		  get_cost(get_predict(feature, w0, w1, b0, b1) - label))
        i += 1
    return w0, w1, b0, b1

def load_data():
	'''
	Desc: 导入数据
	Args: 
	    filename(string) 文件存储位置
    Returns:
        feature_data(mat) 特征
        label_data(mat) 标签
        n_class(int) 类别的个数
	'''

	# 1. 获取特征
	f = open(filename)
	feature_data = []
	label_tmp = []
	for line in f.readlines():
		feature_tmp = []
		lines = line.strip().split('\t')
		for i in range(len(lines)-1):
			feature_tmp.append(float(lines[i]))
		label_tmp.append(int(lines[-1]))
		feature_data.append(feature_tmp)
	f.close()

	# 2.获取标签
	m = len(label_tmp)
	n_class = len(set(label_tmp)) # 类别个数
	label_data = np.mat(np.zeros((m, n_class)))
	for i in range(m):
		label_data[i, label_tmp[i]] = 1
	return np.mat(feature_data), label_data, n_class

def save_model(w0, w1, b0, b1):
	'''
	Desc: 保存最终模型
	Args:
	    w0(mat) 输入层到隐含层之间的权重
	    w1(mat) 隐含层到输出层之间的权重
	    b0(mat) 输入层到隐含层之间的偏置
	    b1(mat) 隐含层到输出层之间的偏置
	'''

	def write_file(file_name, source):
		f = open(file_name, 'w')
		m, n = np.shape(source)
		for i in range(m):
			tmp = []
			for j in range(n):
				tmp.append(str(source[i, j]))
			f.write('\t'.join(tmp) + '\n')
		f.close()
	write_file('weight_w0', w0)
	write_file('weight_w1', w1)
	write_file('weight_b0', b0)
	write_file('weight_b1', b1)

def get_predict(feature, w0, w1, b0, b1):
	'''
	Desc: 计算预测值
	Args:
	    feature(mat) 特征
	    w0(mat) 输入层到隐含层之间的权重
	    w1(mat) 隐含层到输出层之间的权重
	    b0(mat) 输入层到隐含层之间的偏置
	    b1(mat) 隐含层到输出层之间的偏置
	Returns:
	    prediction 预测值
	'''
	hidden_out = hidden_out(hidden_in(feature, w0, b0))
	prediction = predict_out(predict_in(hidden_out, w1, b1))
	return prediction

def err_rate(label, pre):
	'''
	Desc: 计算训练样本上的错误率
	Args:
	    label(mat) 训练样本的标签
	    pre(mat) 训练样本的预测值
	Returns:
	    rate(float) 错误率
	'''

	m = np.shape(label)[0]
	err = .0
	for i in range(m):
		if label[i, 0] != pre[i, 0]:
			err += 1
	rate = err / m
	return rate


if __name__ == '__main__':
	# 1 导入数据
	print('---------- 1. loda data ----------')
	feature, label, n_class = load_data('data.txt')
	# 2 训练网络模型
	print('---------- 2. training ----------')
	w0, w1, b0, b1 = bp_train(feature, label, 20, 1000, 0.1, n_class)
	# 3 保存最终模型
	print('---------- 3. save model ----------')
	save_model(w0, w1, b0, b1)
	# 4 预测结果
	print('---------- 4. get prediction ----------')
	result = get_predict(feature, w0, w1, b0, b1)
	print('训练准确性为：', (1 - err_rate(np.argmax(label, axis=1), np.argmax(result, axis=1))))
