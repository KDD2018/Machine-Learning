import  tensorflow as tf

#使用match_filenames_once()函数获取符合正则表达式的所有文件
#函数原型match_filenames_once(pattern,name)
files = tf.train.match_filenames_once("/home/jiangziyang/TFRecord/data_tfrecords-*")

#通过string_input_producer()函数创建输入队列，输入队列的文件列表是files
#函数原型string_input_producer(string_tensor,num_epochs,shuffle,capacity,
#                                                  shared_name,name,cancel_op)
#函数的shuffle参数用于指定是否随即打乱读文件的顺序，在实际问题中会设置为True
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

#解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        "image_raw":tf.FixedLenFeature([],tf.string),
        "pixels":tf.FixedLenFeature([],tf.int64),
        "label":tf.FixedLenFeature([],tf.int64)
    })
images = tf.decode_raw(features["image_raw"],tf.uint8)
labels = tf.cast(features["label"],tf.int32)
pixels = tf.cast(features["pixels"],tf.int32)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(files))
    #打印文件列表，输出为：
    #[b'/home/jiangziyang/TFRecord/data_tfrecords-1-of-2'
    # b'/home/jiangziyang/TFRecord/data_tfrecords-0-of-2']

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    for i in range(6):
        print(sess.run([images,labels]))

    coordinator.request_stop()
    coordinator.join(threads)