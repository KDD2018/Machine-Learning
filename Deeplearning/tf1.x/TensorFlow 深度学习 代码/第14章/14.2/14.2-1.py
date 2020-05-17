import tensorflow as tf
a = tf.Variable(tf.constant([1.0,2.0],shape=[2]), name="a")
b = tf.Variable(tf.constant([3.0,4.0],shape=[2]), name="b")
result=a+b

init_op=tf.initialize_all_variables()
#设置log_device_placement参数
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init_op)
    print(result)
    #输出Tensor("add:0", shape=(2,), dtype=float32)


'''
仅使用CPU设备，运行会得到以下log：
Device mapping: no known devices.
b: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
b/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
b/Assign: (Assign): /job:localhost/replica:0/task:0/cpu:0
a: (VariableV2): /job:localhost/replica:0/task:0/cpu:0
a/read: (Identity): /job:localhost/replica:0/task:0/cpu:0
add: (Add): /job:localhost/replica:0/task:0/cpu:0
a/Assign: (Assign): /job:localhost/replica:0/task:0/cpu:0
init: (NoOp): /job:localhost/replica:0/task:0/cpu:0
Const_1: (Const): /job:localhost/replica:0/task:0/cpu:0
Const: (Const): /job:localhost/replica:0/task:0/cpu:0
'''