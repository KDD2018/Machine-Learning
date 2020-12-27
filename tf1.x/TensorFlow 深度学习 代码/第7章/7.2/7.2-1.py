import tensorflow as tf
import numpy as np

#使用numpy工具初始化一个名为M的数组，形状为2x3，数据类型为float32
#并使用numpy的reshape()函数调整输入的格式
#注意，M不会被TensorFlow识别为张量
M = np.array([[[2],[1],[2],[-1]],[[0],[-1],[3],[0]],
              [[2],[1],[-1],[4]],[[-2],[0],[-3],[4]]],dtype="float32").reshape(1, 4, 4, 1)

#通过get_variable()函数创建过滤器的权重变量，上面介绍了卷积层
#这里声明的参数变量是一个四维矩阵，前面两个维度代表了过滤器的尺寸，
#第三个维度表示当前层的深度，第四个维度表示过滤器的深度。
filter_weight = tf.get_variable("weights",[2, 2, 1, 1],
    initializer = tf.constant_initializer([[-1, 4],[2, 1]]))

#通过get_variable()函数创建过滤器的偏置项，代码中[1]表示过滤器的深度。
#等于神经网络下一层的深度。
biases = tf.get_variable("biase", [1], initializer = tf.constant_initializer(1))


x = tf.placeholder('float32', [1,None, None,1])

#conv2d()函数实现了卷积层前向传播的算法。
#这个函数的第一个参数为当前层的输入矩阵，注意这个矩阵应该是一个四维矩阵，
#代表第一个维度的参数对应一个输入batch。如果在输入层，input[0, , , ]表示第一张图片，
#input[1, , , ]表示第二张图片，等等。函数第二个参数是卷积层的权重，第三个参数为
#不同维度上的步长。虽然第三个参数提供的是一个长度为4 的数组，
#但是第一个和最后一个数字要求一定是1，这是因为卷积层的步长只对矩阵的长和宽有效。
#最后一个参数是填充(padding的方法，有SAME或VALID 两种选择，
#其中SAME 表示添加全0填充，VALID表示不添加。
#函数原型conv2d(input,filter,strids,padding,us_cudnn_on_gpu_,data_format,name)
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding="SAME")

#bias_add()函数具有给每一个节点加上偏置项点功能。这里不能直接使用加法的原因是
#矩阵上不同位置上的节点都需要加上同样的偏置项。因为过滤器深度为1，
#故偏置项只有一个数，结果为3x4的矩阵中每一个值都要加上这个偏置项。
#原型bias_add(value,bias,data_format,name)
add_bias = tf.nn.bias_add(conv, biases)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    init_op.run()
    M_conv=sess.run(add_bias, feed_dict={x: M})

    #输出结果并不是一个张量，而是数组
    print("M after convolution: \n", M_conv)