import tensorflow as tf

#声明两个变量并计算其加和
a = tf.Variable(tf.constant([1.0,2.0],shape=[2]), name="a")
b = tf.Variable(tf.constant([3.0,4.0],shape=[2]), name="b")
result=a + b

#在声明train.Saver类的同时提供一个列表来指定需要加载的变量a
saver = tf.train.Saver([a])

with tf.Session() as sess:
    #使用restore()函数加载已经保存的模型
    saver.restore(sess, "/home/jiangziyang/model/model.ckpt")
    print(sess.run(a))