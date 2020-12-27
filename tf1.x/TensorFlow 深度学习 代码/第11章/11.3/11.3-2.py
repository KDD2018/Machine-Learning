import tensorflow as tf

#创建先进先出队列
queue = tf.FIFOQueue(100, "float")
#入队操作，每次入队10个随机数值
enqueue = queue.enqueue([tf.random_normal([10])])

#使用QueueRunner创建10个线程进行队列入队操作
#构造函数原型为_init_(self,queue,enqueue_ops,close_op,cancel_op,
#queue_closed_exception_types,queue_runner_def,import_scope)
qr = tf.train.QueueRunner(queue, [enqueue] *10)

# 将定义过的QueueRuner加入到计算图的GraphKeys.QUEUE_RUNERS集合
# 函数原型add_queue_runner(qr,collection)
tf.train.add_queue_runner(qr)

# 定义出队列操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用Coordinator来协同启动的线程
    coordinator = tf.train.Coordinator()

    # 调用start_queue_runners()函数来启动所有线程，并通过参数coord指定
    # 一个Coordinator来处理线程同步终止
    # 函数原型start_queue_runners(sess,coord,daemon,start,collection)
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    #打印全部的100个结果
    for i in range(10):
        print(sess.run(out_tensor))

    #也可以这样定义打印的形式打印每个入队操作的第一个的结果
    #for i in range(10):print(sess.run(out_tensor)[0])

    coordinator.request_stop()
    coordinator.join(threads)

