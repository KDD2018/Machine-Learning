

import time
import numpy as np
import tensorflow as tf
import reader


class PTBModel(object):
    def __init__(self, is_training, config, data, name=None):
        self.batch_size =batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size,
                                                        num_steps, name=name)
        self.size = config.hidden_size
        self.vocab_size = config.vocab_size

        # 使用LSTM结构为循环体结构，并且在训练时使用droppout
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=0.0,
                                             state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=config.keep_prob)

        # 堆叠LSTM 单元lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(config.num_layers)],
                                           state_is_tuple=True)

        # 初始化最初的状态，即全0的向量
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 将单词ID转为单词向量，这里embedding为embedding_lookup()函数的维度信息
        # 单词总数通过vocab_size传入，每个单词向量的维度是size(参数hidden_size的值)，
        # 这样便得出embedding参数的维度
        embedding = tf.get_variable("embedding", [self.vocab_size, self.size],
                                dtype=tf.float32)

        # 通过embedding_lookup()函数将原本batch_size x num_steps个单词ID转为
        # 单词向量，转化后的输入层维度为batch_size x num_steps x size
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 在训练模式下，会对inputs添加dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 定义输出列表，在这里对不同时刻LSTM结构的输出进行汇总，之后会
        # 通过一个全连层得到最终的输出
        outputs = []
        # 定义state存储不同batch中LSTM的状态，并初始化为0
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                # 从输入数据获取当前时刻的输入并传入LSTM结构
                cell_output, state = cell(inputs[:, time_step, :], state)

                # 使用append()函数执行插入操作
                outputs.append(cell_output)

        # concat()函数用于将输出的outputs展开成[batch_size,size*num_steps]的形状
        # 之后再用reshape()函数转为[batch_size*num_steps，size]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.size])
        weight = tf.get_variable("softmax_w", [self.size, self.vocab_size],
                                    dtype=tf.float32)
        bias = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, weight) + bias

        # TensorFlowt提供了legacy_seq2seq.sequence_loss_by_example()函数用于计算
        # 一个序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],[tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        # 计算每个batch的平均损失
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练时定义反向传播操作
        if not is_training:
            return

        self.learning_rate = tf.Variable(0.0, trainable=False)

        # trainable_variables指全部可以训练的参数
        trainable_variables = tf.trainable_variables()

        # 计算self.cost关于trainable_variables的梯度
        gradients = tf.gradients(cost, trainable_variables)

        # 通过clip_by_global_norm()函数控制梯度大小，以免发生梯度膨胀
        clipped_grads, _ = tf.clip_by_global_norm(gradients,config.max_grad_norm)

        # 使用随机梯度下降优化器并定义训练的步骤
        SGDOptimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_op = SGDOptimizer.apply_gradients(zip(clipped_grads, trainable_variables),
                                                     global_step=tf.contrib.framework.\
                                                     get_or_create_global_step())

        self.new_learning_rate = tf.placeholder(tf.float32, shape=[],
                                                name="new_learning_rate")

        self.learning_rate_update = tf.assign(self.learning_rate, self.new_learning_rate)

    # 定义学习率分配函数，该函数会在定义会话时用到
    def assign_lr(self, session, lr_value):
        session.run(self.learning_rate_update, feed_dict={self.new_learning_rate: lr_value})


class Config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    total_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, train_op=None, output_log=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if train_op is not None:
        fetches["train_op"] = train_op

    for step in range(model.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        result = session.run(fetches,feed_dict)

        cost = result["cost"]
        state = result["final_state"]

        costs += cost
        iters += model.num_steps

        if output_log and step % (model.epoch_size//10) == 10:
            print("step%.3f perplexity: %.3f speed: %.0f words/sec" %(step, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


train_data, valid_data, test_data, _ = reader.ptb_raw_data("/home/jiangziyang/PTB/simple-examples/data/")

config = Config()
eval_config = Config()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    # 定义用于训练的循环神经网络模型
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            Model_train = PTBModel(is_training=True, config=config, data=train_data,
                         name="TrainModel")

    # 定义用于验证的循环神经网络模型
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            Model_valid = PTBModel(is_training=False, config=config, data=valid_data,
                              name="ValidModel")

    # 定义用于测试的循环神经网络模型
    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            Model_test = PTBModel(is_training=False, config=eval_config,data=test_data,
                             name="TestModel")

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.total_epoch):
            # 确定学习率衰减，config.max_epoch代表了使用初始学习率的epoch
            # 在这几个epoch内lr_decay会是1
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            Model_train.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(Model_train.learning_rate)))

            # 在所有训练数据上训练循环神经网络模型
            train_perplexity = run_epoch(session, Model_train, train_op=Model_train.train_op,output_log=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            # 使用验证数据评测模型效果
            valid_perplexity = run_epoch(session, Model_valid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # 最后使用测试数据测试模型的效果
        test_perplexity = run_epoch(session, Model_test)
        print("Test Perplexity: %.3f" % test_perplexity)



'''打印的信息
Epoch: 1 Learning rate: 1.000
step10.000 perplexity: 6100.380 speed: 4468 words/sec
step242.000 perplexity: 854.522 speed: 7720 words/sec
step474.000 perplexity: 634.565 speed: 7851 words/sec
step706.000 perplexity: 511.402 speed: 7898 words/sec
step938.000 perplexity: 439.762 speed: 7922 words/sec
step1170.000 perplexity: 393.093 speed: 7938 words/sec
step1402.000 perplexity: 353.468 speed: 7948 words/sec
step1634.000 perplexity: 326.372 speed: 7956 words/sec
step1866.000 perplexity: 304.983 speed: 7961 words/sec
step2098.000 perplexity: 285.320 speed: 7964 words/sec
Epoch: 1 Train Perplexity: 270.662
Epoch: 1 Valid Perplexity: 181.177
Epoch: 2 Learning rate: 1.000
step10.000 perplexity: 209.887 speed: 7911 words/sec
step242.000 perplexity: 150.669 speed: 7996 words/sec
step474.000 perplexity: 158.434 speed: 8001 words/sec
step706.000 perplexity: 153.419 speed: 8000 words/sec
step938.000 perplexity: 150.397 speed: 8001 words/sec
step1170.000 perplexity: 147.956 speed: 8001 words/sec
step1402.000 perplexity: 143.288 speed: 8003 words/sec
step1634.000 perplexity: 141.160 speed: 8003 words/sec
step1866.000 perplexity: 139.159 speed: 8003 words/sec
step2098.000 perplexity: 135.498 speed: 8003 words/sec
Epoch: 2 Train Perplexity: 133.351
Epoch: 2 Valid Perplexity: 142.075
Epoch: 3 Learning rate: 1.000
step10.000 perplexity: 144.825 speed: 7915 words/sec
step242.000 perplexity: 104.702 speed: 8007 words/sec
step474.000 perplexity: 113.971 speed: 8014 words/sec
step706.000 perplexity: 111.025 speed: 8015 words/sec
step938.000 perplexity: 110.071 speed: 8016 words/sec
step1170.000 perplexity: 109.164 speed: 8017 words/sec
step1402.000 perplexity: 106.536 speed: 8018 words/sec
step1634.000 perplexity: 105.908 speed: 8019 words/sec
step1866.000 perplexity: 105.297 speed: 8019 words/sec
step2098.000 perplexity: 103.102 speed: 8017 words/sec
Epoch: 3 Train Perplexity: 102.089
Epoch: 3 Valid Perplexity: 131.913
Epoch: 4 Learning rate: 1.000
step10.000 perplexity: 114.573 speed: 7956 words/sec
step242.000 perplexity: 84.875 speed: 8016 words/sec
step474.000 perplexity: 93.395 speed: 8018 words/sec
step706.000 perplexity: 91.132 speed: 8018 words/sec
step938.000 perplexity: 90.667 speed: 8018 words/sec
step1170.000 perplexity: 90.259 speed: 8019 words/sec
step1402.000 perplexity: 88.330 speed: 8020 words/sec
step1634.000 perplexity: 88.150 speed: 8021 words/sec
step1866.000 perplexity: 87.943 speed: 8020 words/sec
step2098.000 perplexity: 86.353 speed: 8015 words/sec
Epoch: 4 Train Perplexity: 85.770
Epoch: 4 Valid Perplexity: 128.041
Epoch: 5 Learning rate: 0.500
step10.000 perplexity: 99.772 speed: 7972 words/sec
step242.000 perplexity: 70.913 speed: 8001 words/sec
step474.000 perplexity: 76.936 speed: 8002 words/sec
step706.000 perplexity: 74.101 speed: 8002 words/sec
step938.000 perplexity: 73.001 speed: 8002 words/sec
step1170.000 perplexity: 71.967 speed: 8002 words/sec
step1402.000 perplexity: 69.817 speed: 8002 words/sec
step1634.000 perplexity: 69.102 speed: 8003 words/sec
step1866.000 perplexity: 68.329 speed: 8002 words/sec
step2098.000 perplexity: 66.512 speed: 7998 words/sec
Epoch: 5 Train Perplexity: 65.531
Epoch: 5 Valid Perplexity: 119.209
Epoch: 6 Learning rate: 0.250
step10.000 perplexity: 81.382 speed: 7956 words/sec
step242.000 perplexity: 58.566 speed: 7999 words/sec
step474.000 perplexity: 63.781 speed: 8002 words/sec
step706.000 perplexity: 61.303 speed: 8003 words/sec
step938.000 perplexity: 60.323 speed: 8002 words/sec
step1170.000 perplexity: 59.367 speed: 8002 words/sec
step1402.000 perplexity: 57.484 speed: 8002 words/sec
step1634.000 perplexity: 56.770 speed: 8002 words/sec
step1866.000 perplexity: 55.994 speed: 8002 words/sec
step2098.000 perplexity: 54.335 speed: 7999 words/sec
Epoch: 6 Train Perplexity: 53.375
Epoch: 6 Valid Perplexity: 118.528
Epoch: 7 Learning rate: 0.125
step10.000 perplexity: 71.639 speed: 7928 words/sec
step242.000 perplexity: 52.016 speed: 8008 words/sec
step474.000 perplexity: 56.770 speed: 8017 words/sec
step706.000 perplexity: 54.507 speed: 8022 words/sec
step938.000 perplexity: 53.653 speed: 8024 words/sec
step1170.000 perplexity: 52.756 speed: 8026 words/sec
step1402.000 perplexity: 51.042 speed: 8027 words/sec
step1634.000 perplexity: 50.361 speed: 8027 words/sec
step1866.000 perplexity: 49.602 speed: 8027 words/sec
step2098.000 perplexity: 48.050 speed: 8028 words/sec
Epoch: 7 Train Perplexity: 47.137
Epoch: 7 Valid Perplexity: 119.536
Epoch: 8 Learning rate: 0.062
step10.000 perplexity: 67.276 speed: 7964 words/sec
step242.000 perplexity: 48.718 speed: 8030 words/sec
step474.000 perplexity: 53.208 speed: 8033 words/sec
step706.000 perplexity: 51.051 speed: 8033 words/sec
step938.000 perplexity: 50.254 speed: 8033 words/sec
step1170.000 perplexity: 49.405 speed: 8034 words/sec
step1402.000 perplexity: 47.779 speed: 8035 words/sec
step1634.000 perplexity: 47.107 speed: 8034 words/sec
step1866.000 perplexity: 46.360 speed: 8034 words/sec
step2098.000 perplexity: 44.864 speed: 8031 words/sec
Epoch: 8 Train Perplexity: 43.978
Epoch: 8 Valid Perplexity: 120.181
Epoch: 9 Learning rate: 0.031
step10.000 perplexity: 65.039 speed: 7949 words/sec
step242.000 perplexity: 47.037 speed: 7711 words/sec
step474.000 perplexity: 51.398 speed: 7796 words/sec
step706.000 perplexity: 49.286 speed: 7765 words/sec
step938.000 perplexity: 48.500 speed: 7830 words/sec
step1170.000 perplexity: 47.671 speed: 7869 words/sec
step1402.000 perplexity: 46.085 speed: 7896 words/sec
step1634.000 perplexity: 45.417 speed: 7915 words/sec
step1866.000 perplexity: 44.671 speed: 7930 words/sec
step2098.000 perplexity: 43.204 speed: 7941 words/sec
Epoch: 9 Train Perplexity: 42.330
Epoch: 9 Valid Perplexity: 120.289
Epoch: 10 Learning rate: 0.016
step10.000 perplexity: 63.843 speed: 7973 words/sec
step242.000 perplexity: 46.120 speed: 8031 words/sec
step474.000 perplexity: 50.430 speed: 8034 words/sec
step706.000 perplexity: 48.341 speed: 8032 words/sec
step938.000 perplexity: 47.555 speed: 8032 words/sec
step1170.000 perplexity: 46.734 speed: 8033 words/sec
step1402.000 perplexity: 45.167 speed: 8032 words/sec
step1634.000 perplexity: 44.503 speed: 8032 words/sec
step1866.000 perplexity: 43.760 speed: 8032 words/sec
step2098.000 perplexity: 42.310 speed: 8029 words/sec
Epoch: 10 Train Perplexity: 41.441
Epoch: 10 Valid Perplexity: 120.061
Epoch: 11 Learning rate: 0.008
step10.000 perplexity: 63.118 speed: 7977 words/sec
step242.000 perplexity: 45.563 speed: 8028 words/sec
step474.000 perplexity: 49.845 speed: 8031 words/sec
step706.000 perplexity: 47.789 speed: 8031 words/sec
step938.000 perplexity: 47.013 speed: 8032 words/sec
step1170.000 perplexity: 46.200 speed: 8032 words/sec
step1402.000 perplexity: 44.649 speed: 8033 words/sec
step1634.000 perplexity: 43.990 speed: 8033 words/sec
step1866.000 perplexity: 43.252 speed: 8033 words/sec
step2098.000 perplexity: 41.814 speed: 8033 words/sec
Epoch: 11 Train Perplexity: 40.949
Epoch: 11 Valid Perplexity: 119.679
Epoch: 12 Learning rate: 0.004
step10.000 perplexity: 62.666 speed: 7984 words/sec
step242.000 perplexity: 45.215 speed: 8030 words/sec
step474.000 perplexity: 49.483 speed: 8032 words/sec
step706.000 perplexity: 47.455 speed: 8032 words/sec
step938.000 perplexity: 46.693 speed: 8031 words/sec
step1170.000 perplexity: 45.889 speed: 8032 words/sec
step1402.000 perplexity: 44.352 speed: 8032 words/sec
step1634.000 perplexity: 43.698 speed: 8032 words/sec
step1866.000 perplexity: 42.965 speed: 8032 words/sec
step2098.000 perplexity: 41.536 speed: 8033 words/sec
Epoch: 12 Train Perplexity: 40.675
Epoch: 12 Valid Perplexity: 119.350
Epoch: 13 Learning rate: 0.002
step10.000 perplexity: 62.386 speed: 7967 words/sec
step242.000 perplexity: 45.007 speed: 8027 words/sec
step474.000 perplexity: 49.268 speed: 8031 words/sec
step706.000 perplexity: 47.261 speed: 8032 words/sec
step938.000 perplexity: 46.509 speed: 8030 words/sec
step1170.000 perplexity: 45.715 speed: 8031 words/sec
step1402.000 perplexity: 44.187 speed: 8032 words/sec
step1634.000 perplexity: 43.537 speed: 8033 words/sec
step1866.000 perplexity: 42.807 speed: 8033 words/sec
step2098.000 perplexity: 41.383 speed: 8031 words/sec
Epoch: 13 Train Perplexity: 40.273
Epoch: 13 Valid Perplexity: 117.392
Test Perplexity: 113.702                                             
'''