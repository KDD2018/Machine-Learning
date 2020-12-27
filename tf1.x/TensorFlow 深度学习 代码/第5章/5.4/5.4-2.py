import tensorflow as tf
import  numpy as np

#定义训练轮数
training_steps = 30000

#定义输入的数据和对应的标签并在for循环内进行填充
data = []
label = []
for i in range(200):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)
    #这里对产生的x1和x2进行判断，如果产生的点落在半径为1的圆内，则label值为0
    #否则label值为1
    if x1**2 + x2**2 <= 1:
        data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

#numpy的hstack()函数用于在水平方向将元素堆起来
#函数原型numpy.hstack(tup)，参数tup可以是元组、列表、或者numpy数组，
#返回结果为numpy的数组。reshape()函数的参数-1表示行列进行翻转。
#这样处理的结果为data变成了200x2大小的数组，而label是100x1
data = np.hstack(data).reshape(-1,2)
label = np.hstack(label).reshape(-1, 1)

#定义完成前向传播的隐层
def hidden_layer(input_tensor,weight1,bias1,weight2,bias2,weight3,bias3):
    layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+bias1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
    return tf.matmul(layer2,weight3)+bias3


x = tf.placeholder(tf.float32, shape=(None, 2),name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1),name="y-output")

#定义权重参数和偏置参数
weight1 = tf.Variable(tf.truncated_normal([2,10],stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
weight2 = tf.Variable(tf.truncated_normal([10,10],stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
weight3 = tf.Variable(tf.truncated_normal([10,1],stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1]))

#用len()函数计算data数组的长度
sample_size = len(data)

#得到隐层前向传播结果
y= hidden_layer(x,weight1,bias1,weight2,bias2,weight3,bias3)

#自定义的损失函数。pow()函数用于计算幂函数，原型为pow(x,y,name=None)
#返回结果为x的y次幂，这里返回结果为(y_-y)^2用于衡量计算值与实际值的差距
error_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
tf.add_to_collection("losses", error_loss)      #加入集合的操作

#在权重参数上实现L2正则化
regularizer = tf.contrib.layers.l2_regularizer(0.01)
regularization = regularizer(weight1)+regularizer(weight2)+regularizer(weight3)
tf.add_to_collection("losses",regularization)     #加入集合的操作

#get_collection()函数获取指定集合中的所有个体，这里是获取所有损失值
#并在add_n()函数中进行加和运算
loss = tf.add_n(tf.get_collection("losses"))

#定义一个优化器，学习率为固定为0.01，注意在实际应用中这个学习率数值应该大于0.01
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #在for循环内进行30000训练
    for i in range(training_steps):
        sess.run(train_op, feed_dict={x: data, y_: label})

        #训练30000轮，但每隔2000轮就输出一次loss的值
        if i % 2000 == 0:
            loss_value = sess.run(loss, feed_dict={x: data, y_: label})
            print("After %d steps, loss_value is: %f" % (i,loss_value))

