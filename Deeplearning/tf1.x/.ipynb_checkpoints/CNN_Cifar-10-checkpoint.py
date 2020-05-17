import Cifar10_data
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 4000
batch_size = 128
num_examples_for_eval = 10000
data_dir = "/home/jiangziyang/Cifar10_data/cifar-10-batches-bin"


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:

        # multiply()函数原型multiply(x,y,name)
        # l2_loss()函数原型l2_loss(t,name)
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return var

#对于用于训练的图片数据，distorted参数为True，表示进行数据增强处理
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir,
                                         batch_size=batch_size,distorted=True)

#对于用于训练的图片数据，distorted参数为Nnone，表示不进行数据增强处理
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir,
                                       batch_size=batch_size,distorted=None)


#创建placeholder
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])


#第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME")


#第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME")

# 拉直数据
# reshape()函数原型reshape(tensor,shape,name)
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value


#第一个全连层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)


#第二个全连层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)


#第三个全连层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)


#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
                                                labels=tf.cast(y_, tf.int64))
weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#函数原型in_top_k(predictions,targets,k,name)
top_k_op = tf.nn.in_top_k(result, y_, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    #开启多线程
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch,
                                                              y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            #打印每一轮训练的耗时
            print("step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"%
                     (step, loss_value, examples_per_sec, sec_per_batch))


    #math.ceil()函数用于求整，原型为ceil(x)
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size

    # 在一个for循环内统计所有预测正确的样例的个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch,
                                                      y_: label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%" % ((true_count/total_sample_count)*100))

'''打印的内容
step 0, loss = 4.68 (24.0 examples/sec; 5.330 sec/batch)
step 100, loss = 2.07 (785.0 examples/sec; 0.163 sec/batch)
step 200, loss = 1.81 (1252.3 examples/sec; 0.102 sec/batch)
step 300, loss = 1.66 (1177.9 examples/sec; 0.109 sec/batch)
step 400, loss = 1.57 (1204.8 examples/sec; 0.106 sec/batch)
step 500, loss = 1.54 (1151.7 examples/sec; 0.111 sec/batch)
step 600, loss = 1.38 (1163.0 examples/sec; 0.110 sec/batch)
step 700, loss = 1.24 (1237.4 examples/sec; 0.103 sec/batch)
step 800, loss = 1.42 (1135.9 examples/sec; 0.113 sec/batch)
step 900, loss = 1.27 (1227.6 examples/sec; 0.104 sec/batch)
step 1000, loss = 1.48 (1162.3 examples/sec; 0.110 sec/batch)
step 1100, loss = 1.35 (1161.9 examples/sec; 0.110 sec/batch)
step 1200, loss = 1.35 (1208.6 examples/sec; 0.106 sec/batch)
step 1300, loss = 1.21 (1220.0 examples/sec; 0.105 sec/batch)
step 1400, loss = 1.17 (1182.8 examples/sec; 0.108 sec/batch)
step 1500, loss = 1.29 (1030.8 examples/sec; 0.124 sec/batch)
step 1600, loss = 1.36 (1114.4 examples/sec; 0.115 sec/batch)
step 1700, loss = 1.08 (1182.8 examples/sec; 0.108 sec/batch)
step 1800, loss = 1.23 (1130.6 examples/sec; 0.113 sec/batch)
step 1900, loss = 1.10 (1119.0 examples/sec; 0.114 sec/batch)
step 2000, loss = 1.35 (1088.9 examples/sec; 0.118 sec/batch)
step 2100, loss = 1.22 (1172.3 examples/sec; 0.109 sec/batch)
step 2200, loss = 1.17 (1137.9 examples/sec; 0.112 sec/batch)
step 2300, loss = 1.16 (1114.5 examples/sec; 0.116 sec/batch)
step 2400, loss = 1.05 (1104.1 examples/sec; 0.116 sec/batch)
step 2500, loss = 1.01 (1100.6 examples/sec; 0.116 sec/batch)
step 2600, loss = 1.06 (1173.9 examples/sec; 0.109 sec/batch)
step 2700, loss = 1.16 (1155.2 examples/sec; 0.111 sec/batch)
step 2800, loss = 1.07 (1114.5 examples/sec; 0.115 sec/batch)
step 2900, loss = 1.06 (1150.9 examples/sec; 0.111 sec/batch)
step 3000, loss = 1.35 (1808.9 examples/sec; 0.118 sec/batch)
step 3100, loss = 1.12 (1170.3 examples/sec; 0.109 sec/batch)
step 3200, loss = 1.27 (1133.9 examples/sec; 0.112 sec/batch)
step 3300, loss = 1.06 (1150.6 examples/sec; 0.111 sec/batch)
step 3400, loss = 1.05 (1104.1 examples/sec; 0.116 sec/batch)
step 3500, loss = 1.11 (1104.5 examples/sec; 0.116 sec/batch)
step 3600, loss = 1.03 (1173.2 examples/sec; 0.109 sec/batch)
step 3700, loss = 1.10 (1145.2 examples/sec; 0.116 sec/batch)
step 3800, loss = 1.07 (1104.5 examples/sec; 0.116 sec/batch)
step 3900, loss = 1.06 (1140.9 examples/sec; 0.111 sec/batch)
accuracy = 74.6%
'''