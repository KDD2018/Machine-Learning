import tensorflow as tf
import tensorflow.contrib.slim as slim

#用slim的srg_scope()函数设置一些会用到卷积或池化函数的默认参数，
#这包括stride=1和padding="SAME"
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,
                                                            padding="SAME"):

    #在这里为InceptionModule创建一个统一的变量命名空间,模块里面有多个路径，
    #每一条路径都会接收模块之前网络节点的输出，这里用last_net统一代表这个输出
    with tf.variable_scope("Module"):

        #使用变量空间的名字标识模块的路径，这类似于"BRANCH_n"的形式，
        #例如BRANCH_1表示这个模块里的第二条路径
        with tf.variable_scope("BRANCH_0"):
            branch_0 = slim.conv2d(last_net,320,[1,1],scope="Conv2d_0a_1x1")

        with tf.variable_scope("BRANCH_1"):
            branch_1 = slim.conv2d(last_net,384,[1,1],scope="Conv2d_1a_1x1")
            #concat()函数实现了拼接的功能，函数原型为concat(values,axis,name)
            #第一个参数用于指定拼接的维度信息，对于InceptionModule，值一般为3，
            #表示在第三个维度上进行拼接(串联)，第二个参数是用于拼接的两个结果
            branch_1 = tf.concat(3,[
                slim.conv2d(branch_1, 384, [1,3], scope="Conv2d_1b_1x3"),
                slim.conv2d(branch_1, 384, [3,1], scope="Conv2d_1c_3x1")])

        with tf.variable_scope("BRANCH_2"):
            branch_2 = slim.conv2d(last_net,448,[1,1],scope="Conv2d_2a_1x1")
            branch_2 = slim.conv2d(branch_2, 384, [3,3], scope="Conv2d_2b_3x3")
            branch_2 = tf.concat(3,[
                slim.conv2d(branch_2, 384, [1,3], scope="Conv2d_2c_1x3"),
                slim.conv2d(branch_2, 384, [3,1], scope="Conv2d_2d_3x1")])

        with tf.variable_scope("BRANCH_3"):
            branch_3 = slim.avg_pool2d(last_net, [3,3], scope="AvgPool_3a_3x3")
            branch_2 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_3a_1x1")

        #最后用concat()函数将InceptionModule每一条路径的结果进行拼接得到最终结果
        Module_output = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])