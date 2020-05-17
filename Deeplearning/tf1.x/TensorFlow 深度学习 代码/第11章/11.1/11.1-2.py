import tensorflow as tf


#创建一个TFRecordReader类的实例
reader = tf.TFRecordReader()

#创建一个队列对输入文件列表进行维护，队列的知识放到了本章的稍后
#函数原型string_input_producer(string_tensor,num_epochs,shuffle,seed,
#                                  capacity,shared_name,name,cancel_op)
filename_queue = tf.train.string_input_producer(
                       ["/home/jiangziyang/TFRecord/MNIST_tfrecords"])

#使用TFRecordReader.read()函数从文件中读取一个样例，原型reader(self,queue,name)
#也可使用read_up_to()函数一次性读取多个样例，
#原型read_up_to(self,queue,num_records,name)
_,serialized_example = reader.read(filename_queue)

#使用parse_single_example()函数解析读取的样例。
#原型parse_single_example(serialized,features,name,example_names)
features = tf.parse_single_example(
    serialized_example,
    features={
        #可以使用FixedLenFeature类对属性进行解析，
        "image_raw":tf.FixedLenFeature([],tf.string),
        "pixels":tf.FixedLenFeature([],tf.int64),
        "label":tf.FixedLenFeature([],tf.int64)
    })

#decode_raw()函数用于将字符串解析成图像对应的像素数组
#函数原型decode_raw(bytes,out_type,little_endian,name)
images = tf.decode_raw(features["image_raw"],tf.uint8)
#使用cast()函数进行类型转换
labels = tf.cast(features["label"],tf.int32)
pixels = tf.cast(features["pixels"],tf.int32)


with tf.Session() as sess:
    #启动多线程处理输入数据，多线程处理数据也会在本章的稍后予以介绍
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    for i in range(10):
        image, label, pixel = sess.run([images, labels, pixels])
        print(label)
        #输出7 3 4 6 1 8 1 0 9 8

