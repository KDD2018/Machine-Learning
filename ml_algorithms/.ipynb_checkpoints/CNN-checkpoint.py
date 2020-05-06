#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np


class CNN():
    def __init__(self, input_data_trX, input_data_trY, input_data_vaX, input_data_vaY, input_data_teX, input_data_teY):
        # 第一个卷积层的权重和偏置
        self.w = None
        self.b = None
        # 第二个卷积层的权重和偏置
        self.w2 = None
        self.b2 = None
        # 第三个卷积层的权重和偏置
        self.w3 = None
        self.b3 = None
        # 全连接层的权重和偏置
        self.W4 = None
        self.b4 = None
        # 隐含层到输出层的权重和偏置
        self.w_o = None
        self.b_o = None
        # 卷积层中样本保持不变的比例
        self.p_keep_conv = None
        # 全连接层中样本保持不变的比例
        self.p_keep_hidden = None
        # 训练集、验证集和测试集的特征和标签
        self.trX = input_data_trX
        self.trY = input_data_trY
        self.vaX = input_data_vaX
        self.vaY = input_data_vaY
        self.teX = input_data_teX
        self.teY = input_data_teY
        
    def fit(self):
        X = tf.placeholder('float', [None, 28, 28, 1])
        Y = tf.placeholder('float', [None, 10])
        
        # 第一层32个3*3的卷积核
        self.w = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        self.b = tf.Variable(tf.constant(0.0, shape=[32]))
        # 第二层为64个3*3的卷积核
        self.w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        self.b2 = tf.Variable(tf.constant(0.0, shape=[64]))
        # 第三层为128个3*3的卷积核
        self.w2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        self.b2 = tf.Variable(tf.constant(0.0, shape=[128]))
        # 全连接层