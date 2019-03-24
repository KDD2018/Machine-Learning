#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

'''
卷积层一：
		卷积: 32个Filter，5*5， strides 1, padding='SAME'    输入：[None, 28, 28, 1]		输出: [None, 28, 28, 32]    bias=32
		激活： [None, 28, 28, 32]
		池化: 2*2, strides 2, padding='SAME'	[None, 28, 28, 32]---------->[None, 14, 14, 32]
卷积层二：
		卷积：64个Filter，5*5， strides 1, padding='SAME'		输入： [None, 14, 14, 32]   输出： [None, 14, 14, 64]  bias=64
		激活：[None, 14, 14, 64]
		池化：2*2, strides 2  padding='SAME'  [None, 14, 14, 64]----------->[None, 7, 7, 64]
全连接层：
		[None, 7*7*64]  [7*7*64, 10]  [None, 10]   bias=10
'''


def conv_fc():



	return None


if __name__ == '__main__':
	conv_fc()

