import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/jiangziyang/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, weight) + bias
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 准备验证数据
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(30000):
        if i % 1000 == 0:
            # 计算滑动平均模型在验证数据上的结果。
            # 为了能得到百分数输出，需要将得到的validate_accuracy扩大100倍
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%" % (i, validate_accuracy * 100))

        xs, ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After 30000 trainging step(s) ,test accuracy using average"
          " model is %g%%" % (test_accuracy * 100))

'''打印的信息
After 0 trainging step(s) ,validation accuracyusing average model is 9.58%
After 1000 trainging step(s) ,validation accuracyusing average model is 89.58%
After 2000 trainging step(s) ,validation accuracyusing average model is 86.36%
After 3000 trainging step(s) ,validation accuracyusing average model is 88.98%
After 4000 trainging step(s) ,validation accuracyusing average model is 90.74%
After 5000 trainging step(s) ,validation accuracyusing average model is 90.3%
After 6000 trainging step(s) ,validation accuracyusing average model is 89.1%
After 7000 trainging step(s) ,validation accuracyusing average model is 90.4%
After 8000 trainging step(s) ,validation accuracyusing average model is 91.12%
After 9000 trainging step(s) ,validation accuracyusing average model is 88.94%
After 10000 trainging step(s) ,validation accuracyusing average model is 89.08%
After 11000 trainging step(s) ,validation accuracyusing average model is 90.06%
After 12000 trainging step(s) ,validation accuracyusing average model is 91.26%
After 13000 trainging step(s) ,validation accuracyusing average model is 89.54%
After 14000 trainging step(s) ,validation accuracyusing average model is 89.98%
After 15000 trainging step(s) ,validation accuracyusing average model is 90.82%
After 16000 trainging step(s) ,validation accuracyusing average model is 87.74%
After 17000 trainging step(s) ,validation accuracyusing average model is 89.22%
After 18000 trainging step(s) ,validation accuracyusing average model is 90.94%
After 19000 trainging step(s) ,validation accuracyusing average model is 90.18%
After 20000 trainging step(s) ,validation accuracyusing average model is 91%
After 21000 trainging step(s) ,validation accuracyusing average model is 89.76%
After 22000 trainging step(s) ,validation accuracyusing average model is 90.12%
After 23000 trainging step(s) ,validation accuracyusing average model is 91.02%
After 24000 trainging step(s) ,validation accuracyusing average model is 91.44%
After 25000 trainging step(s) ,validation accuracyusing average model is 89.42%
After 26000 trainging step(s) ,validation accuracyusing average model is 90.28%
After 27000 trainging step(s) ,validation accuracyusing average model is 89.34%
After 28000 trainging step(s) ,validation accuracyusing average model is 89.68%
After 29000 trainging step(s) ,validation accuracyusing average model is 91.5%
After 30000 trainging step(s) ,test accuracy using average model is 91.13% 
'''