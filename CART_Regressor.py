#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np


class Node:
	'''
	Desc: 数的节点的类
	'''
	def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
		self.fea = fea  # 用于切分数据集的属性的列索引
		self.value = value  # 划分的点
		self.results = results  # 存储叶节点的值
		self.right = right  # 右子树
		self.left = left  # 左子树


def load_data(data_file):
	'''
	Desc: 导入训练数据
	Args: datafile(string) 保存训练数据的文件
	Returns: data(list) 训练数据
	'''
	data = []
	f = open(data_file)
	for line in f.readlines():
		sample = []
		lines = line.strip().split('\t')
		for x in lines:
			sample.append(float(x))
		data.append(sample)
	f.close()
	return data

def err_cnt(dataSet):
	'''
	Desc: 回归树的划分指标
	Args: dataSet(list) 训练数据
	Returns: m*s^2(float) 总方差
	'''
	data = np.mat(dataSet)
	return np.var(data[:, -1]) * np.shape(data)[0]

def split_tree(data, fea, value):
	'''
	Desc: 根据特征fea中的值value将数据集data划分成左右子树
	Args: data(list) 训练样本
		  fea(float) 需要划分的特征index
		  value(float) 切分点
	Returns: (set_1, set_2)(tuple) 左右子树的聚合
	'''
	set_1 = []
	set_2 = []
	for x in data:
		if x[fea] > value:
			set_1.append(x)
		else:
			set_2.append(x)
	return (set_1, set_2)

def leaf(dataSet):
	'''
	Desc: 计算叶节点的值
	Args: dataSets(list) 训练样本
	Returns: np.mean(data[:, -1])(float) 均值
	'''
	data = np.mat(dataSet)
	return np.mean(data[:, -1])


def build_tree(data, min_sample, min_err):
	'''
	Desc: 构建回归树
	Args: data(list)训练样本
		  min_sample(int) 叶子节点中最少的样本数
		 min_err(float) 最小的error
	Returns: node 树的根节点
	'''
	# 构建决策树，函数返回决策树的根节点
	if len(data) <= min_sample:
		return Node(results=leaf(data))

	# 1. 初始化
	best_err = err_cnt(data)
	bestCriteria = None # 存储最佳切分属性以及最佳切分点
	bestSets = None # 存储切分后的两个数据

	# 2. 开始构建CART回归树
	feature_num = len(data[0]) - 1
	for fea in range(0, feature_num):
		feature_values = {}
		for sample in data:
			feature_values[sample[fea]] = 1

		for value  in feature_values.keys():
			# 2.1 尝试划分
			(set_1, set_2) = split_tree(data, fea, value)
			if len(set_1) < 2 or len(set_2) < 2:
				continue
			# 2.2 计算划分后的error
			now_err = err_cnt(set_1) + err_cnt(set_2)
			# 2.3 更新最有划分
			if now_err < best_err and len(set_1) > 0 and len(set_2) >0:
				best_err = now_err
				bestCriteria = (fea, value)
				bestSets = (set_1, set_2)
	# 3. 判断划分是否结束
	if best_err > min_err:
		right = build_tree(best_Sets[0], min_sample, min_err)
		left = build_tree(best_Sets[1], min_sample, min_err)
		return Node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
	else:
		return Node(results=leaf(data)) # 返回当前类别标签作为最终的分类标签


if __name__ == "__main__":
	# 1. 导入训练集
	print('------------- 1.load data -------------')
	data = load_data('sine.txt')
	# 2. 构建CART树
	print('------------- 2.build CART tree ---------------')
	regression_tree = build_tree(data, 30, 0.3)
	# 3. 评估CART树
	print('------------- 3.cal_err --------------')
	err = cal_err(data, regression_tree)