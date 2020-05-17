# 使用BasicLSTMCell类定义一个LSTM结构
# 假设BasicLSTMCell类已从core_rnn_cell_impl中被导入
lstm = BasicLSTMCell(lstm hidden size)

# 将LSTM中的状态初始化为全0 数组
# BasicLSTMCell类提供了zero_state()函数来生成全0的初始状态
# 对于循环神经网络每次使用的一个batch的训练样本，以下代码中，batch_size就是这个
# batch 的大小
state = lstm.zero_state(batch_size,tf.float32)

# 定义loss用于存储损失值
loss = 0.0

# 用for循环模拟RNN的循环过程，由于循环神经网络存在梯度弥散的问题
# 所以一般会规定循环的长度，在这里num_steps就是这个循环的长度
for i in range (num_steps):

    if i = 0:
        tf.get_variable_scope().reuse_variables()

    # 进行LSTM处理，传递到lstm中的current_input是当前时刻的输入，state是前一时刻状态
    # 处理后可以得到当前lstm的输出lstm_output和更新后的状态state。
    1stm_output,state = lstm(current_input, state)

    # 用函数fc()代表一个全连层
    # 这里将当前时刻lstm的输出传入一个全连接层得到最后的输出。
    output = fc(lstm_output)

    # 计算当前时刻输出的损失
    # calculate_loss()函数代表计算损失的操作，其中expected_output是期望的输出
    loss += calculate_loss(output,expected_output)


