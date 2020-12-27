import tensorflow as tf

#在变量空间外部创建变量a
a = tf.get_variable("a",[1],initializer=tf.constant_initializer(1.0))
print(a.name)
#输出a:0，"a"是这个变量的名称，":0"表示这个变量生成变量的第一个结果

with tf.variable_scope("one"):
    a2 = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
    print(a2.name)
    #输出one/a:0，其中one表示a所属的变量空间为one，

with tf.variable_scope("one"):
    with tf.variable_scope("two"):
        a4 = tf.get_variable("a", [1])
        print(a4.name)
        #输出one/two/a:0，变量空间嵌套之后，
        #变量名的名称会加入所有变量空间的名称作为前缀

    b = tf.get_variable("b", [1])
    print(b.name)
    #输出one/b:0，因为退出了变量空间two

with tf.variable_scope("",reuse=True):
    #也可以直接通过带变量空间名称前缀的变量名来获取相应的变量
    a5 = tf.get_variable("one/two/a",[1])
    print(a5 == a4)
    #输出True

