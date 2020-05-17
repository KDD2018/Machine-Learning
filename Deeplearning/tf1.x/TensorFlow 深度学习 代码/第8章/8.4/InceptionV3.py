import tensorflow as tf
import os
import flower_photos_dispose as fd
from tensorflow.python.platform import gfile

model_path = "/home/jiangziyang/InceptionModel/inception_dec_2015/"
model_file = "tensorflow_inception_graph.pb"

num_steps = 4000
BATCH_SIZE = 100

bottleneck_size = 2048  # InceptionV3模型瓶颈层的节点个数

# 调用create_image_lists()函数获得该函数返回的字典
image_lists = fd.create_image_dict()
num_classes = len(image_lists.keys())  # num_classes=5,因为有5类

# 读取已经训练好的Inception-v3模型。
with gfile.FastGFile(os.path.join(model_path, model_file), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 使用import_graph_def()函数加载读取的InceptionV3模型后会返回
# 图像数据输入节点的张量名称以及计算瓶颈结果所对应的张量，函数原型为
# import_graph_def(graph_def,input_map,return_elements,name,op_dict,producer_op_list)
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,
                                                          return_elements=["pool_3/_reshape:0",
                                                                           "DecodeJpeg/contents:0"])

x = tf.placeholder(tf.float32, [None, bottleneck_size], name='BottleneckInputPlaceholder')
y_ = tf.placeholder(tf.float32, [None, num_classes], name='GroundTruthInput')

# 定义一层全连接层
with tf.name_scope("final_training_ops"):
    weights = tf.Variable(tf.truncated_normal([bottleneck_size, num_classes], stddev=0.001))
    biases = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(x, weights) + biases
    final_tensor = tf.nn.softmax(logits)

# 定义交叉熵损失函数以及train_step使用的随机梯度下降优化器
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)

# 定义计算正确率的操作
correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(y_, 1))
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_steps):
        # 使用get_random_bottlenecks()函数产生训练用的随机的特征向量数据及其对应的label
        # 在run()函数内开始训练的过程
        train_bottlenecks, train_labels = fd.get_random_bottlenecks(sess, num_classes,
                                                                    image_lists, BATCH_SIZE,
                                                                    "training",
                                                                    jpeg_data_tensor, bottleneck_tensor)
        sess.run(train_step, feed_dict={x: train_bottlenecks, y_: train_labels})

        # 进行相关的验证，同样是使用get_random_bottlenecks()函数产生随机的特征向量及其
        # 对应的label
        if i % 100 == 0:
            validation_bottlenecks, validation_labels = fd.get_random_bottlenecks(sess,
                                                                                  num_classes, image_lists,
                                                                                  BATCH_SIZE, "validation",
                                                                                  jpeg_data_tensor, bottleneck_tensor)
            validation_accuracy = sess.run(evaluation_step, feed_dict={
                x: validation_bottlenecks,
                y_: validation_labels})
            print("Step %d: Validation accuracy = %.1f%%" % (i, validation_accuracy * 100))

    # 在最后的测试数据上测试正确率，这里调用的是get_test_bottlenecks()函数，返回
    # 所有图片的特征向量作为特征数据
    test_bottlenecks, test_labels = fd.get_test_bottlenecks(sess, image_lists, num_classes,
                                                            jpeg_data_tensor, bottleneck_tensor)
    test_accuracy = sess.run(evaluation_step, feed_dict={x: test_bottlenecks,
                                                         y_: test_labels})
    print("Finally test accuracy = %.1f%%" % (test_accuracy * 100))

'''
Step 0: Validation accuracy = 46.0%
Step 100: Validation accuracy = 81.0%
Step 200: Validation accuracy = 93.0%
Step 300: Validation accuracy = 89.0%
Step 400: Validation accuracy = 83.0%
Step 500: Validation accuracy = 84.0%
Step 600: Validation accuracy = 90.0%
Step 700: Validation accuracy = 90.0%
Step 800: Validation accuracy = 93.0%
Step 900: Validation accuracy = 91.0%
Step 1000: Validation accuracy = 87.0%
Step 1100: Validation accuracy = 92.0%
Step 1200: Validation accuracy = 93.0%
Step 1300: Validation accuracy = 93.0%
Step 1400: Validation accuracy = 84.0%
Step 1500: Validation accuracy = 92.0%
Step 1600: Validation accuracy = 90.0%
Step 1700: Validation accuracy = 83.0%
Step 1800: Validation accuracy = 89.0%
Step 1900: Validation accuracy = 95.0%
Step 2000: Validation accuracy = 91.0%
Step 2100: Validation accuracy = 91.0%
Step 2200: Validation accuracy = 94.0%
Step 2300: Validation accuracy = 93.0%
Step 2400: Validation accuracy = 90.0%
Step 2500: Validation accuracy = 94.0%
Step 2600: Validation accuracy = 94.0%
Step 2700: Validation accuracy = 88.0%
Step 2800: Validation accuracy = 93.0%
Step 2900: Validation accuracy = 88.0%
Step 3000: Validation accuracy = 88.0%
Step 3100: Validation accuracy = 95.0%
Step 3200: Validation accuracy = 95.0%
Step 3300: Validation accuracy = 93.0%
Step 3400: Validation accuracy = 90.0%
Step 3500: Validation accuracy = 86.0%
Step 3600: Validation accuracy = 93.0%
Step 3700: Validation accuracy = 89.0%
Step 3800: Validation accuracy = 96.0%
Step 3900: Validation accuracy = 95.0%
Finally test accuracy = 94.6%
'''