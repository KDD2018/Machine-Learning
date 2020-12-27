import tensorflow as tf

a = tf.placeholder(tf.float32,shape=(2),name="input")
b = tf.placeholder(tf.float32,shape=(2),name="input")
result = a+b

with tf.Session() as sess:
    sess.run(result,feed_dict={a:[1.0,2.0]})
    #没有提供b的值所以报错，报错信息：
    #InvalidArgumentError (see above for traceback): You must feed
    #a value for placeholder tensor 'input_1' with dtype float and shape [2]
    print(result)

