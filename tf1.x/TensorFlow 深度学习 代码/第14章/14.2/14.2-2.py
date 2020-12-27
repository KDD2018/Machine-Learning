

import tensorflow as tf

#通过device()函数将运算指定到CPU设备上。
with tf.device("/cpu:0"):
    a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a")
    b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b")

#通过device()函数将运算指定到第一个GPU设备上。
with tf.device("/gpu:0"):
    result = a + b

#log_device_placement参数可以用来记录运行每一个运算的设备。
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.initialize_all_variables().run()
    print(sess.run(result))
    #输出Tensor("add:0", shape=(2,), dtype=float32)
'''
指定运算设备后的log输出：
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0,name: GeForce GTX 850M, pci bus id: 000:01:00.0
/job:localhost/replica:0/task:0/device:XLA_GPU:0 device:XLA_GPU device
/job:localhost/replica:0/task:0/device:XLA_CPU:0 device:XLA_CPU device
b: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
b/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
b/Assign: (Assign): /job:localhost/replica:0/task:0/cpu:0
a: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
a/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
add: (Add): /job:localhost/replica:0/task:0/gpu:0
a/Assign: (Assign): /job:localhost/replica:0/task :0/cpu:0
init: (No0p): /job:localhost/replica:0/task:0/cpu:0
Const_1: (Const): /job:localhost/replica:0/task:0/cpu:0
Const: (Const): /job:localhost/replica:0/task:0/cpu:0
'''
