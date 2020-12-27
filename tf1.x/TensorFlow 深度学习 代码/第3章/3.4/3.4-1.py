import tensorflow as tf

#用placeholder定义一个位置
#原型placeholder(dtype,shape,name)
a = tf.placeholder(tf.float32,shape=(2),name="input")
b = tf.placeholder(tf.float32,shape=(2),name="input")
result = a+b

with tf.Session() as sess:
    #Session.run()函数有很多参数，原型为
    #sess.run(self,fetches,feed_dict,options,run_metadata)
    #fetches参数接受result，feed_dict参数指定了需要提供的值
    sess.run(result,feed_dict={a:[1.0,2.0],b:[3.0,4.0]})
    print(result)
    #输出Tensor("add:0", shape=(2,), dtype=float32)
