import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def write_to_tfrecord(mnist_data_path, tfrecord_path):
    '''
    将数据写入tfrecord文件
    :param mnist_data_path: 源数据路径
    :param tfrecord_path: tfrecord文件路径
    :return: None
    '''
    # 读取MNIST数据
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
    print(mnist.train.images.shape)

    # 创建TFRecord文件存储器
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # 循环将数据填充到Example协议内存块，并写入TFRecord文件
    for i in range(mnist.train.num_examples):
        # 将图像矩阵转成一个字符串
        image_to_string = mnist.train.images[i].tostring()

        # 构造一个样本的Example协议块
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_to_string])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.argmax(mnist.train.labels[i])]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # 将一个Example协议块序列化，并写入TFRecord
        writer.write(example.SerializeToString())

    # 关闭存储器
    writer.close()


def read_tfrecord(tfrecord_path):
    '''
    读取tfrecord文件
    :param tfrecord_path: 
    :return: images, labels
    '''
    # 创建队列
    filename_queue = tf.train.string_input_producer([tfrecord_path])

    # 建立tfrecord文件阅读器
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    # 解析Example协议内存块
    features_dict = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(value, features=features_dict)

    # 解码内容，解码字符串为图像对应的图像像素数组(string需要解码)
    image = tf.decode_raw(features['image'], tf.uint8)

    # 转换label数据类型
    label = tf.cast(features['label'], tf.int32)



    return image, label




if __name__ == '__main__':

    # 读取数据并写入tfrecord文件
    # write_to_tfrecord('./data/MNIST_DATA/', './data/TFRecord')

    # 读取tfrecord文件
    image, label = read_tfrecord('./data/TFRecord')
    # 开启会话，执行读取过程
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        for i in range(5):
            # 循环读取执行5次
            images, labels = sess.run([image, label])

        # 回收子线程
        coord.request_stop()
        coord.join(threads)




