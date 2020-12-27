import tensorflow as tf

with tf.variable_scope("one"):
    # 使用get_variable_scope()函数可以获取当前的变量空间
    print(tf.get_variable_scope().reuse)
    # 输出False

    with tf.variable_scope("two", reuse=True):
        # 通过上下文管理器新建一个嵌套的变量空间，并指定reuse=True
        print(tf.get_variable_scope().reuse)
        # 输出True

        # 在一个嵌套的变量空间中如果不指定reuse参数，
        # 那么会默认为和外面最近的一层保持一致
        with tf.variable_scope("three"):
            print(tf.get_variable_scope().reuse)
            # 输出True

    # 回到reuse值为默认为False的最外层的变量空间
    print(tf.get_variable_scope().reuse)
    # False
