import tensorflow as tf
weights = tf.constant([[1.0,2.0],[-3.0,-4.0]])

#regularizer_l2是l2_regularizer()函数返回的函数
regularizer_l2 = tf.contrib.layers.l2_regularizer(.5)

#regularizer_l1是l1_regularizer()函数返回的函数
regularizer_l1 = tf.contrib.layers.l1_regularizer(.5)
with tf.Session() as sess:
    print(sess.run(regularizer_l2(weights)))
    #输出7.5
    print(sess.run(regularizer_l1(weights)))
    #输出5.0