import tensorflow as tf

with tf.variable_scope("one"):
    a = tf.get_variable("var1", [1])
    print(a.name)
    #输出one/var1:0

with tf.variable_scope("two"):
    b = tf.get_variable("var2", [1])
    print(b.name)
    #输出two/var2:0


with tf.name_scope("a"):
    #使用Variable()函数生成变量会受到name_scope()的影响
    #主要表现为在变量名称前添加变量空间前缀
    a = tf.Variable([1],name="a")
    print(a.name)
    #a/a:0

    #使用get_variable()函数生成变量不会受到name_scope()的影响
    #主要表现为在变量名称前不会添加变量空间前缀
    a = tf.get_variable("b", [1])
    print(a.name)
    #输出b:0

with tf.name_scope("b"):
    #企图创建name属性为b的变量c，然而这个变量已经被声明了
    #所以会被报错
    #ValueError: Variable b already exists, disallowed.
    #Did you mean to set reuse=True in VarScope?
    c = tf.get_variable("b",[1])
