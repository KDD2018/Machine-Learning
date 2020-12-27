import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import MultiRNNCell

#使用LSTM作为循环体结构
lstm = BasicLSTMCell(lstm_size)

#使用 MultiRNNCell类实现深层循环神经网络的前向传播过程，在构造类实例时，
#参数number_of_laryers表示同一时刻的循环神经网络有多少层
stacked_lstm = MultiRNNCell([lstm] * number_of_laryers)

#定义初始状态
state = stacked_lstm.zero_state(batch_size, tf.float32)

for i in range(num_steps):
    if i>0:
        tf.get_variable_scope().reuse_variables()
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fc(stacked_lstm_output)
    loss +=calculate_loss(final_output, expexted_output)