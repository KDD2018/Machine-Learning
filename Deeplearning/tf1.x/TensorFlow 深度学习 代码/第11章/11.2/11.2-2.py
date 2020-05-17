import  tensorflow as tf
import  numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/jiangziyang/MNIST_data",
                                             dtype=tf.uint8,one_hot=True)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images = mnist.test.images
labels = mnist.test.labels
pixels = images.shape[1]
num_examples = mnist.test.num_examples

#num_files定义总共写入多少个文件
num_files = 2

for i in range(num_files):
    #将数据写入多个文件时，为区分这些文件可以添加后缀
    filename = ("/home/jiangziyang/TFRecord/data_tfrecords-%.1d-of-%.1d"
                                                         % (i, num_files))
    writer = tf.python_io.TFRecordWriter(filename)

    #将Example结构写入TFRecord文件，写入文件的过程和11.1节一样。
    for index in range(num_examples):
        image_string = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            "pixels": _int64_feature(pixels),
            "label": _int64_feature(np.argmax(labels[index])),
            "image_raw": _bytes_feature(image_string)
        }))
        writer.write(example.SerializeToString())
    writer.close()