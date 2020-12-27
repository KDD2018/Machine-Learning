import tensorflow as tf

#声明两个变量并计算其加和
a = tf.Variable(tf.constant([1.0,2.0],shape=[2]), name="a")
b = tf.Variable(tf.constant([3.0,4.0],shape=[2]), name="b")
result=a+b

#定义Saver类对象用于保存模型
saver=tf.train.Saver()

with tf.Session() as sess:
    # 使用restore()函数加载已经保存的模型
    saver.restore(sess,"/home/jiangziyang/model/model.ckpt")
    print(sess.run(result))
    # 输出为[4. 6.]
    # restore函数的原型是restore(self,sess,save_path)
