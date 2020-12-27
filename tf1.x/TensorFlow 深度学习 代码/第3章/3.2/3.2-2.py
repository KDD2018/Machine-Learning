import tensorflow as tf

# 使用Graph()函数创建一个计算图
g1 = tf.Graph()
with g1.as_default():  # 将定义的计算图使用as_default()函数设为默认

    # 创建计算图中的变量并设置初始值为
    a = tf.get_variable("a", [2], initializer=tf.ones_initializer())
    b = tf.get_variable("b", [2], initializer=tf.zeros_initializer())

# 使用Graph()函数创建另一个计算图
g2 = tf.Graph()
with g2.as_default():
    a = tf.get_variable("a", [2], initializer=tf.zeros_initializer())
    b = tf.get_variable("b", [2], initializer=tf.ones_initializer())

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))
        # 打印[1. 1.]
        #    [0. 0.]

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))
        # 打印[0. 0.]
        #    [1. 1.]