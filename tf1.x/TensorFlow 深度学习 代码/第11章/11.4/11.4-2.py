import tensorflow as tf

files = tf.train.match_filenames_once("/home/jiangziyang/TFRecord/data_tfrecords-*")
filename_queue = tf.train.string_input_producer(files, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        "image_raw": tf.FixedLenFeature([], tf.string),
        "pixels": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64)
    })
images = tf.decode_raw(features["image_raw"], tf.uint8)
labels = tf.cast(features["label"], tf.int32)
pixels = tf.cast(features["pixels"], tf.int32)

batch_size = 10

capacity = 5000 + 3 * batch_size

images.set_shape(784)

# 使用shuffle_batch()函数在组织样例数据成batch之前将样例数据顺序打乱，
# 对于min_after_dequeue参数，假设是100
min_after_dequeue = 100
image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                  batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):
        xs, ys = sess.run([image_batch, label_batch])
        print(xs, ys)

    coord.request_stop()
    coord.join(threads)

'''打印的内容
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] [3 9 4 1 9 6 9 3 5 4]
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] [9 4 3 7 0 1 1 3 7 0]
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] [6 1 7 3 1 4 5 8 3 7]
'''