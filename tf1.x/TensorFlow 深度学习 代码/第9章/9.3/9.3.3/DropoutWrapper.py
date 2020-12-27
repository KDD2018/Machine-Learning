from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import DropoutWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import MultiRNNCell

#定义LSTM结构
lstm = BasicLSTMCell(lstm_size)

#使用DropoutWrapper实现循环体的Dropout功能
#DropoutWrapper类构造函数原型__init__(self,cell,input_keep_prob,output_keep_prob,seed)
dropout_lstm = DropoutWrapper(lstm,output_keep_prob=0.5)

#使用MultiRNNCell在深度方向堆叠循环体结构
stacked_lstm = MultiRNNCell([dropout_lstm] * number_of_layers)