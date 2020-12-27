import tensorflow as tf
a = tf.Variable(0, dtype=tf.float32, name="a")
b = tf.Variable(0, dtype=tf.float32, name="b")

#在使用字典方式给变量赋值时可以不用定义滑动平均类
averages_class = tf.train.ExponentialMovingAverage(0.99)

#使用字典的方式给变量a和b赋值
saver = tf.train.Saver({"a/ExponentialMovingAverage":a,"b/ExponentialMovingAverage":b})

#也可以使用train.ExponentialMovingAverage类提供的variables_to_restore()函数
#直接生成上面代码中提供的字典，所以下面这一句和上面那一句效果相同
#函数原型variables_to_restore(self,moving_avg_variable)
#saver = tf.train.Saver(averages_class.variables_to_restore())

with tf.Session() as sess:

    saver.restore(sess,"/home/jiangziyang/model/model2.ckpt")

    print(sess.run([a, b]))
    # 输出结果为
    #[0.9561783, 0.47808915]

    print(averages_class.variables_to_restore())
    # 输出结果为
    #{'a/ExponentialMovingAverage': < tensorflow.python.ops.variables.Variable
    #object at 0x7f35bff4ab70 >,
    #'b/ExponentialMovingAverage': < tensorflow.python.ops.variables.Variabl
    #object at 0x7f35b19b7dd8 >}
