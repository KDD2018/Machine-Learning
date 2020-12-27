import tensorflow as tf

#通过device()函数将运算指定到GPU设备上。
#注意这里a和b不再是浮点值，而是整数
with tf.device("/gpu:0"):
    a = tf.Variable(tf.constant([1, 2], shape=[2]), name="a")
    b = tf.Variable(tf.constant([3, 4], shape=[2]), name="b")
result = a + b
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.initialize_all_variables().run()
    print(sess.run(result))

'''
输出以下报错信息：
InvalidArgumentError (see above for traceback): Cannot assign a device to node 'b': 
Could not satisfy explicit device specification '/device:GPU:0' because no supported 
kernel for GPU devices is available.
Colocation Debug Info:
Colocation group had the following types and devices: 
Assign: CPU XLA_CPU XLA_GPU 
Identity: CPU XLA_CPU XLA_GPU 
VariableV2: CPU XLA_CPU XLA_GPU 
	 [[Node: b = VariableV2[container="", dtype=DT_INT32, shape=[2], shared_name="",
	                                                      _device="/device:GPU:0"]()]]
'''