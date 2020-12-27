import tensorflow as tf

files = tf.train.match_filenames_once("/home/jiangziyang/TFRecord/data_tfrecords-*")
#创建文件队列shuffle参数设置为True，打乱文件
filename_queue = tf.train.string_input_producer(files, shuffle=True)

#实例化TFRecordReader类，准备读取TFRecord文件
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

#解析读取的样例，这部分和11.1.1节是相同的
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

#设置每个batch中样例的个数
batch_size = 10

#用于组合成batch的队列中最多可以缓存的样例的个数
capacity = 5000 + 3 * batch_size

#set_shape()函数用来设置设置尺寸，这一步操作也可以使用image.resize_images()
#函数来完成。设置尺寸的操作是必须的，在1.0.0版的TensorFlow中如果没有这一步
#会在batch()函数的capacity参数处报错ValueError: All shapes must be fully
#defined: [TensorShape([Dimension(None)]), TensorShape([])]
images.set_shape(784)

#使用batch()函数将样例组合成batch
#函数原型batch(tensors,batch_size,num_threads,capacity,enqueue_many,shapes,
#                        dynamic_pad,allow_smaller_final_batch,shared_name,name)
image_batch, label_batch = tf.train.batch([images, labels],
                                    batch_size=batch_size,capacity=capacity,)



#shuffle_batch()函数会在组织样例数据成batch之前将数据顺序打乱，
#这两个函数的使用情况类似。shuffle_batch()函数原型为
#shuffle_batch(tensors,batch_size,capacity,min_after_dequeue,num_threads,
#                seed,enqueue_many,shapes,allow_smaller_final_batch,shared_name,name)
#min_after_dequeue=5000
#image_batch, label_batch = tf.train.shuffle_batch([images, labels],
#                                    batch_size=batch_size,capacity=capacity,
#                                                  min_after_dequeue=min_after_dequeue)

#接下来就可以像之前的网络模型那样创建会话并开始训练了，与之前的训练过程相比，
#这里的会话中增加了多线程处理的相关代码
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #一般在这个循环内开始训练，这里设定了训练轮数为3
    #在每一轮训练的过程中都会执行一个组合样例为batch的操作并打印出来
    for i in range(3):
        xs,ys=sess.run([image_batch,label_batch])
        print(xs,ys)

    coord.request_stop()
    coord.join(threads)