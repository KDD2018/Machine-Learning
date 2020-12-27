import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
batch_size = 100
data_dir = "/home/jiangziyang/MNIST_data"
log_dir = "/home/jiangziyang/log"
mnist = input_data.read_data_sets(data_dir, one_hot=True)


def variable_summaries(var):
    with tf.name_scope("summaries"):
        # 求解函数传递进来的var参数的平均值，并使用scaler()函数进行汇总
        # 函数scalar()原型为scalar(name,tensor,collections)
        # 其中参数name是展示在ＴensorBoard上的标签，tensor就是要汇总的数据
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        # 汇总var数据的方差值,并将标签设为stddev
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)

        # 汇总var数据的最大值
        tf.summary.scalar("max", tf.reduce_max(var))

        # 汇总var数据的最小值
        tf.summary.scalar("min", tf.reduce_min(var))

        # 使用histogram()将var数据汇总为直方图的形式
        # 函数原型histogram(name,values,collections)
        # 其中参数name是展示在ＴensorBoard上的标签，tensor就是要汇总的数据
        tf.summary.histogram("histogram", var)


def create_layer(input_tensor, input_num, output_num, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # 创建权重参数，并调用variable_summaries()方法统计权重参数的最大、最小
            # 均值、方差等信息
            weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.1))
            variable_summaries(weights)

        with tf.name_scope("biases"):
            # 创建偏偏置参数，并调用variable_summaries()方法统计偏置参数的最大、最小
            # 均值、方差等信息
            biases = tf.Variable(tf.constant(0.1, shape=[output_num]))
            variable_summaries(biases)

        with tf.name_scope("Wx_add_b"):
            # 计算没有加入激活的线性变换的结果，并通过histogram()函数汇总为直方图数据
            pre_activate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram("pre_activations", pre_activate)

        # 计算激活后的线性变换的结果，并通过histogram()函数汇总为直方图数据
        activations = act(pre_activate, name="activation")
        tf.summary.histogram("activations", activations)

        return activations

x = tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

hidden_1 = create_layer(x, 784, 500, "layer_1")
y = create_layer(hidden_1, 500, 10, "layer_y", act=tf.identity)

with tf.name_scope("input_reshape"):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image("input", image_shaped_input, 10)

# 计算交叉熵损失并汇总为标量数据
with tf.name_scope("cross_entropy"):
    cross = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy = tf.reduce_mean(cross)
    tf.summary.scalar("cross_entropy_scalar", cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#计算预测精度并汇总为标量数据
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy_scalar", accuracy)

# 使用merge_all()函数直接获取所有汇总操作
merged = tf.summary.merge_all()

saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 训练过程，测试过程
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test")

    # 测试过程的feed数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):

        # 运行测试过程并输出日志文件到log下的test目录下
        if i % 100 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)
            test_writer.add_summary(summary, i)
            print("Accuracy at step %s,accuracy is: %s%%" % (i, acc * 100))

        # 产生训练数据，运行训练过程
        else:
            x_train, y_train = mnist.train.next_batch(batch_size=batch_size)
            if i % 100 == 50:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict={x: x_train, y_: y_train},
                                      options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step%03d" % i)
                train_writer.add_summary(summary, i)

                #注意，这里保存模型不是为了后期使用，而是为了可视化降维后的嵌入向量
                saver.save(sess, log_dir+"/model.ckpt",i)

                print("Adding run metadata for", i)
            else:

                summary, _ = sess.run([merged, train_step], feed_dict={x: x_train, y_: y_train})
                train_writer.add_summary(summary, i)

    # 关闭ＦileWriter
    train_writer.close()
    test_writer.close()

'''打印的信息
Accuracy at step 0,accuracy is: 9.34000015258789%
Adding run metadata for 50
Accuracy at step 100,accuracy is: 91.44999980926514%
Adding run metadata for 150
Accuracy at step 200,accuracy is: 93.43000054359436%
Adding run metadata for 250
Accuracy at step 300,accuracy is: 94.16000247001648%
Adding run metadata for 350
Accuracy at step 400,accuracy is: 95.03999948501587%
Adding run metadata for 450
Accuracy at step 500,accuracy is: 95.38999795913696%
Adding run metadata for 550
Accuracy at step 600,accuracy is: 95.92999815940857%
Adding run metadata for 650
Accuracy at step 700,accuracy is: 96.14999890327454%
Adding run metadata for 750
Accuracy at step 800,accuracy is: 96.3100016117096%
Adding run metadata for 850
Accuracy at step 900,accuracy is: 96.71000242233276%
Adding run metadata for 950
'''