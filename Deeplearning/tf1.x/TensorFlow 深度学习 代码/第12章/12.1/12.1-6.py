import tensorflow as tf
a = tf.Variable(0, dtype=tf.float32, name="a")
b = tf.Variable(0, dtype=tf.float32, name="b")

# 定义滑动平均操作
averages_class = tf.train.ExponentialMovingAverage(0.99)
averages_op = averages_class.apply(tf.all_variables())

# 输出这个计算图所有的变量，这些变量在集合tf.GraphKeys.VARIABLES下
for variables in tf.global_variables():
    print(variables.name)
    # 输出结果为
    # a:0
    # b:0
    # a/ExponentialMovingAverage:0
    # b/ExponentialMovingAverage:0

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)

    # 使用assign()函数对变量值进行更新
    # 原型为assign(ref,value,validate_shape,ues_locking,name)
    sess.run(tf.assign(a, 10))
    sess.run(tf.assign(b, 5))

    # 执行滑动平均操作
    sess.run(averages_op)

    saver.save(sess, "/home/jiangziyang/model/model2.ckpt")

    print(sess.run([a, averages_class.average(a)]))
    print(sess.run([b, averages_class.average(b)]))
    # 输出结果为
    # [10.0, 0.099999905]
    # [5.0, 0.049999952]