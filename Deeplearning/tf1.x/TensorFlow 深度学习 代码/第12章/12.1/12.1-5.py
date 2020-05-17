import tensorflow as tf

# 声明两个变量，但是名称和已经保存的模型中的变量名称不同
a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a2")
b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b2")
result =a+b

#saver=tf.train.Saver()
saver=tf.train.Saver({"a":a,"b":b})

with tf.Session() as sess:
    saver.restore(sess,"/home/jiangziyang/model/model.ckpt")
    print(sess.run(result))
    ##输出[4. 6.]