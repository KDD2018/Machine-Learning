import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

# 生成一个写日志的writer，并将当前TensorFlow计算图写入日志。
# 原型__init__(self,logdir,graph,max_queue,flush_secs,graph_def)
# 参数logdir就是日志文件所在的路径，而参数graph就是需要写入日志的计算图
writer = tf.summary.FileWriter("/home/jiangziyang/log", tf.get_default_graph())
writer.close()

