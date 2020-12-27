import tensorflow as tf

# 通过device()函数将运算指定到GPU设备上。
with tf.device("/gpu:0"):
    a = tf.Variable(tf.constant([1, 2], shape=[2]), name="a")
    b = tf.Variable(tf.constant([3, 4], shape=[2]), name="b")
result = a + b
with tf.Session(config=tf.ConfigProto(log_device_placement=True,
							              allow_soft_placement=True)) as sess:
    tf.initialize_all_variables().run()
    print(sess.run(result))
	#输出[4  6]

'''
经过llow_soft_placement=True之后，log会输出以下结果：
Device mapping:
/job:localhost/replica:0/task:0/gpu:0->device: 0,name: GeForce GTX 850M,pci bus id: 000:01:00.0
/job:localhost/replica:0/task:0/device:XLA_GPU:0 device:XLA_GPU device
/job:localhost/replica:0/task:0/device:XLA_CPU:0 device:XLA_CPU device
b: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
b/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
b/Assign: (Assign): /job:localhost/replica:0/task:0/cpu:0
a: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
a/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
add: (Add): /job:localhost/replica:0/task:0/gpu:0
a/Assign: (Assign): /job:localhost/replica:0/task :0/cpu:0
init: (No0p): /job:localhost/replica:0/task:0/gpu:0
Const_1: (Const): /job:localhost/replica:0/task:0/gpu:0
Const: (Const): /job:localhost/replica:0/task:0/gpu:0
'''
