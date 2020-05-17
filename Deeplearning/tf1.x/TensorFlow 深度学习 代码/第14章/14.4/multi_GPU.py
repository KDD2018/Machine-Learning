import tensorflow as tf
from datetime import datetime
import time

# 定义训练神经网络时需要用到的参数。
batch_size = 100
learning_rate_base = 0.001
learning_rate_decay = 0.99
num_steps = 1000

n_GPU = 3  # 定义使用到的GPU的数量
log_dir = "/home/jiangziyang/log/"  # 定义日志输出的路径


# 定义函数inference()实现了模型的前向传播过程
def inference(input_tensor):
    # L2正则化项
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    with tf.variable_scope("layer_1"):
        weights = tf.get_variable("weights", [784, 500], initializer=tf. \
                                  truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(weights))
        biases = tf.get_variable("biases", [500], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer_y"):
        weights = tf.get_variable("weights", [500, 10], initializer=tf. \
                                  truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularizer(weights))
        biases = tf.get_variable("biases", [10], initializer=tf.constant_initializer(0.0))
        layery = tf.matmul(layer1, weights) + biases
    return layery


# 通过get_data()函数从输入队列中得到训练用的数据，关于TFRecord文件的读取可参考第十一章
def get_data():
    filename_queue = tf.train.string_input_producer(
                      ["/home/jiangziyang/TFRecord/MNIST_tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "pixels": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    decoded_image = tf.decode_raw(features["image_raw"], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [784])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features["label"], tf.int32)
    capacity = 10000 + 3 * batch_size
    x, y_ = tf.train.shuffle_batch([retyped_image, label], batch_size=batch_size,
                                   capacity=capacity, min_after_dequeue=10000)
    return x, y_


# 定义tower_loss()函数计算损失，包括交叉熵损失和L2正则化损失
def tower_loss(x, y_, scope, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = inference(x)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                                  labels=y_))
    regularization_loss = tf.add_n(tf.get_collection("losses", scope))
    loss = cross_entropy + regularization_loss
    return loss


# 定义average_gradients()函数计算每一个变量经由多个GPU计算得到的梯度的平均值。
def average_gradients(tower_grads):
    average_grads = []

    # 通过枚举的方式获得所有变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):

        # 求解所有GPU上计算得到的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        # 将计算得到的平均梯度放到列表中，并且和变量是一一对应的
        average_grads.append(grad_and_var)

    return average_grads


with tf.Graph().as_default():
    x, y_ = get_data()

    global_step = tf.get_variable("global_step", [],
                                  initializer=tf.constant_initializer(0), trainable=False)

    # 指数衰减的学习率
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                               60000 / batch_size, learning_rate_decay)

    SGDOptimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # 计算损失值及梯度值的任务被放在几个单独的GPU上，
    # 列表tower_grads用于存储每个GPU计算得到的梯度的值
    tower_grads = []
    reuse_variables = False
    for i in range(n_GPU):
        with tf.device("/gpu:%d" % i):
            with tf.name_scope("GPU_%d" % i) as scope:
                # 计算损失值
                loss = tower_loss(x, y_, scope, reuse_variables)
                reuse_variables = True

                # 计算梯度值
                grads = SGDOptimizer.compute_gradients(loss)
                tower_grads.append(grads)

    # 调用average_gradients()函数计算变量的平均梯度。
    grads = average_gradients(tower_grads)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

    # 使用平均梯度更新参数。
    apply_gradient_op = SGDOptimizer.apply_gradients(grads, global_step=global_step)

    # 使用TensorBoard统计直方图的方式展示可训练变量的取值
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 计算变量的滑动平均值。
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = apply_gradient_op = tf.no_op("train")

    summary_op = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True)) as sess:

        # 初始化所有变量并启动队列。
        tf.initialize_all_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for step in range(num_steps):

            # 执行神经网络训练操作，并记录训练操作的运行时间。
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            # 每隔100步展示当前的训练进度和loss值，并通过summary_op执行所有的汇总操作。
            if step != 0 and step % 100 == 0:
                examples_per_sec = (batch_size * n_GPU) / duration
                sec_per_batch = duration / n_GPU

                print("%s: step %d, %.1f examples/sec and %.3f sec/batch,loss = %.2f" %
                      (datetime.now(), step, examples_per_sec, sec_per_batch, loss_value))

                summary = sess.run(summary_op)
                summary_writer.add_summary(summary, step)

        coord.request_stop()
        coord.join(threads)

'''打印的信息
2018-05-20 17:04:44.918390: step 100, 3870.0 examples/sec and 0.026 sec/batch,loss = 20.28
2018-05-20 17:04:52.430348: step 200, 4143.6 examples/sec and 0.024 sec/batch,loss = 8.87
2018-05-20 17:05:00.058975: step 300, 3950.4 examples/sec and 0.025 sec/batch,loss = 2.87
2018-05-20 17:05:07.822249: step 400, 3753.7 examples/sec and 0.027 sec/batch,loss = 2.22
2018-05-20 17:05:15.672554: step 500, 4058.2 examples/sec and 0.025 sec/batch,loss = 0.89
2018-05-20 17:05:23.499220: step 600, 4239.6 examples/sec and 0.024 sec/batch,loss = 2.80
2018-05-20 17:05:31.143852: step 700, 3745.5 examples/sec and 0.027 sec/batch,loss = 2.54
2018-05-20 17:05:38.912934: step 800, 4447.4 examples/sec and 0.022 sec/batch,loss = 1.94
2018-05-20 17:05:46.736062: step 900, 4152.1 examples/sec and 0.024 sec/batch,loss = 1.58
'''
