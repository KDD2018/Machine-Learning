import tensorflow as tf

x = tf.constant([0.9, 0.85], shape=[1, 2])

# 使用随机正态分布函数声明w1和w2两个变量，其中w1是2x3的矩阵，w2是3x1的矩阵
# 这里使用了随机种子参数seed，这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")

# 将biase(偏置项)参数b1设置为初始值全为0的1x3矩阵，b2是初始值全为1的1x1矩阵
b1 = tf.Variable(tf.zeros([1, 3]))
b2 = tf.Variable(tf.ones([1]))

init_op = tf.global_variables_initializer()

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
    # 输出[[5.4224963]]