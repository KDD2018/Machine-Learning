import tensorflow as tf

# 在参与矩阵相乘运算时，需要通过shape参数直接指定矩阵形状
# 或者指定value参数时通过[ ]的方式间接指定形状，比如[[0.9,0.85]]
x = tf.constant([0.9, 0.85], shape=[1, 2])

# 使用常数生成函数声明w1和w2两个变量作为weight参数
# 这里也通过shape参数指定了矩阵的形状，w1为2x3的矩阵，w2为3x1的矩阵
w1 = tf.Variable(tf.constant([[0.2, 0.1, 0.3], [0.2, 0.4, 0.3]], shape=[2, 3]), name="w1")
w2 = tf.Variable(tf.constant([0.2, 0.5, 0.25], shape=[3, 1]), name="w2")

# b1和b2作为biase(偏置项)参数
b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name="b1")
b2 = tf.constant([-0.3], shape=[1], name="b2")

# 初始化全部变量
# 也可以使用initialize_all_variables()函数，这在稍早的TensorFlow版本中很常见
init_op = tf.global_variables_initializer()

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2
# matmul()函数的原型是
# matmul(a,b,transpose_a,transpose_b,adjoint_a,adjoint_b,a_is_spare,b_is_spare,name)

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
    # 最后输出[[0.15625]]