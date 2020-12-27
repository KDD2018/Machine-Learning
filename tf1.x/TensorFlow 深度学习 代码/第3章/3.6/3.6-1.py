import tensorflow as tf

# 在名字为one的命名空间内创建名字为a的变量
with tf.variable_scope("one"):
    a = tf.get_variable("a", [1],initializer = tf.constant_initializer(1.0))

    #因为在命名空间one内已经存在名字为a的变量，所以下面的这段代码将被报错
with tf.variable_scope("one"):
    a2 = tf.get_variable("a", [1])
    #报错信息
    #ValueError: Variable one/a already exists, disallowed. Did you mean to set 
    #reuse=True in VarScope? Originally defined at: 