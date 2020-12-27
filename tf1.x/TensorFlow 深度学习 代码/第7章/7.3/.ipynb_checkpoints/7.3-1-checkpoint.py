import tensorflow as tf
import numpy as np

M = np.array([[[-2],[2],[0],[3]],
              [[1],[2],[-1],[2]],
              [[0],[-1],[1],[0]]],dtype="float32").reshape(1, 3, 4, 1)
filter_weight = tf.get_variable("weights",[2, 2, 1, 1],
    initializer = tf.constant_initializer([[2, 0],[-1, 1]]))
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))
x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
add_bias = tf.nn.bias_add(conv, biases)

#max_pool()函数实现了最大池化层的前向传播过程
#原型为max_pool(value,strides,padding,data_format,name)
#参数value为输入数据，strides为提供了步长信息，padding提供了是否使用全0填充。
pool = tf.nn.max_pool(add_bias, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    M_conv = sess.run(add_bias, feed_dict={x: M})
    M_pool = sess.run(pool, feed_dict={x: M})
    print(" after average pooled: \n", M_pool)
    '''输出内容
    after average pooled:
    [[[[7.]
       [5.]]
      [[1.]
       [3.]]]]
    '''