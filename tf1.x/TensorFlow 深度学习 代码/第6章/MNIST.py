
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/jiangziyang/MNIST_data",one_hot=True)

batch_size = 100                #设置每一轮训练的batch大小
learning_rate = 0.8             #学习率
learning_rate_decay = 0.999     #学习率的衰减
max_steps = 30000               #最大训练步数

#定义存储训练轮数的变量，在使用Tensorflow训练神经网络时，
#一般会将代表训练轮数的变量通过trainable参数设置为不可训练的
training_step = tf.Variable(0,trainable=False)

#定义得到隐藏层和输出层的前向传播计算方式，激活函数使用relu()
def hidden_layer(input_tensor,weights1,biases1,weights2,biases2,layer_name):
    layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    return tf.matmul(layer1,weights2)+biases2

x = tf.placeholder(tf.float32,[None,784],name="x-input")   #INPUT_NODE=784
y_ = tf.placeholder(tf.float32,[None,10],name="y-output")   #OUT_PUT=10
#生成隐藏层参数，其中weights包含784x500=392000个参数
weights1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1,shape=[500]))
#生成输出层参数，其中weights2包含500x10=5000个参数
weights2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[10]))

#计算经过神经网络前向传播后得到的y的值，这里没有使用滑动平均
y = hidden_layer(x,weights1,biases1,weights2,biases2,'y')


#初始化一个滑动平均类，衰减率为0.99
#为了使模型在训练前期可以更新地更快，这里提供了num_updates参数
#并设置为当前网络的训练轮数
averages_class = tf.train.ExponentialMovingAverage(0.99,training_step)
#定义一个更新变量滑动平均值的操作需要向滑动平均类的apply()函数提供一个参数列表
#train_variables()函数返回集合图上Graph.TRAINABLE_VARIABLES中的元素，
#这个集合的元素就是所有没有指定trainable_variables=False的参数
averages_op = averages_class.apply(tf.trainable_variables())
#再次计算经过神经网络前向传播后得到的y的值，这里使用了滑动平均，但要牢记滑动平均值只是一个影子变量
average_y = hidden_layer(x,averages_class.average(weights1),averages_class.average(biases1),
                averages_class.average(weights2),averages_class.average(biases2),'average_y')

#计算交叉熵损失的函数原型为sparse_softmax_cross_entropy_with_logits(_sential, labels, logdits, name)
#它与softmax_cross_entropy_with_logits()函数的计算方式相同，适用于每个类别相互独立且排斥的情况，
#即一幅图只能属于一类。在1.0.0版本的TensorFlow中，这个函数只能通过命名参数的方式来使用，在这里
#logits参数是神经网络不包括softmax层的前向传播结果，lables参数给出了训练数据的正确答案
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
#argmax()函数原型为argmax(input, axis, name, dimension),用于计算每一个样例的预测答案，
#其中input参数y是一个batch_size * 10(batch_size行，10列)的二维数组，每一行表示一个样例前向传播的结果，
#axis参数“1”表示选取最大值的操作仅在第一个维度中进行，即只在每一行选取最大值对应的下标。
#于是得到的结果是一个长度为batch_size的一维数组，这个一维数组中的值就表示了每一个样例对应的
#数字识别结果。


regularizer = tf.contrib.layers.l2_regularizer(0.0001)       #计算L2正则化损失函数
regularization = regularizer(weights1)+regularizer(weights2) #计算模型的正则化损失
loss = tf.reduce_mean(cross_entropy)+regularization          #总损失


#用指数衰减法设置学习率，这里staircase参数采用默认的False，即学习率连续衰减
laerning_rate = tf.train.exponential_decay(learning_rate,training_step,mnist.train.num_examples/batch_size,
                                                                                       learning_rate_decay)
#使用GradientDescentOptimizer优化算法来优化交叉熵损失和正则化损失
train_step= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=training_step)


#在训练这个模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数又需要
#更新每一个参数的滑动平均值，control_dependencies()用于完成这样的一次性多次操作，
# 同样的操作也可以使用下面这行代码完成：
# train_op = tf.group(train_step,averages_op)
with tf.control_dependencies([train_step,averages_op]):
     train_op = tf.no_op(name="train")


#检查使用了滑动平均值模型的神经网络前向传播结果是否正确。
#equal()函数原型为equal(x, y, name)，用于判断两个张量的每一维是否相等，如果相等返回True,否则返回False。
crorent_predicition = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

#cast()函数原型为cast(x, DstT, name)。在这里用于将一个bool型的数据转为float32类型
#之后会将得到的float32 的数据求一个平均值，这个平均值就是模型在这一组数据上的正确率
accuracy = tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))


with tf.Session() as sess:
    #在稍早的版本中一般使用 initialize_all_variables()函数初始化全部变量
    tf.global_variables_initializer().run()

    #准备验证数据，
    validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
    #准备测试数据，
    test_feed = {x:mnist.test.images,y_:mnist.test.labels}

    for i in range(max_steps):
        if i%1000==0:
            #计算滑动平均模型在验证数据上的结果。
            # 为了能得到百分数输出，需要将得到的validate_accuracy扩大100倍
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d trainging step(s) ,validation accuracy"
                  "using average model is %g%%"%(i,validate_accuracy*100))

        #产生这一轮使用的一个batch的训练数据，并进行训练
        #input_data.read_data_sets()函数生成的类提供了train.next_bacth()函数，
        #通过设置函数的batch_size参数就可以从所有的训练数据中提读取一小部分作为一个训练batch
        xs,ys = mnist.train.next_batch(batch_size=100)
        sess.run(train_op,feed_dict={x:xs,y_:ys})

    #使用测试数据集检验神经网络训练之后的最终正确率
    # 为了能得到百分数输出，需要将得到的test_accuracy扩大100倍
    test_accuracy = sess.run(accuracy,feed_dict=test_feed)
    print("After %d trainging step(s) ,test accuracy using average"
                  " model is %g%%"%(max_steps,test_accuracy*100))
