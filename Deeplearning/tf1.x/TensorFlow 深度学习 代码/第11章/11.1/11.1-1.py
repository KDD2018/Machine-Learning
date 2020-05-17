import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/jiangziyang/MNIST_data",
                                  dtype=tf.uint8, one_hot=True)


# 定义生成整数型和字符串型属性的方法，这是将数据填入到Example协议内存块
# (protocol buffer)的第一步，以后会调用到这个方法
def Int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def Bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 读取mnist数据。
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

#输出TFRecord文件的地址(相对于系统根目录)。
filename = "/home/jiangziyang/TFRecord/MNIST_tfrecords"

# 创建一个python_io.TFRecordWriter()类的实例
writer = tf.python_io.TFRecordWriter(filename)

# for循环执行了将数据填入到Example协议内存块的主要操作
for i in range(num_examples):
    # 将图像矩阵转化成一个字符串
    image_to_string = images[i].tostring()

    feature = {
        "pixels": Int64_feature(pixels),
        "label": Int64_feature(np.argmax(labels[i])),
        "image_raw": Bytes_feature(image_to_string)
    }
    features = tf.train.Features(feature=feature)

    # 定义一个Example，将相关信息写入到这个数据结构
    example = tf.train.Example(features=features)

    # 将一个Example写入到TFRecord文件
    # 原型writer(self, record)
    writer.write(example.SerializeToString())

# 在写完文件后最好的习惯是调用close()函数关闭
writer.close()