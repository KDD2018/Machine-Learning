import  tensorflow as tf

#创建FIFOQueue先进先出队列，同时指定队列中可以保存的元素个数以及数据类型
#原型_init_(self,capacity,dtypes,shared_name,names,shapes,name)
Queue = tf.FIFOQueue(2, "int32")

#使用FIFOQueue类的enqueue_many()函数初始化队列中的元素
#和初始化变量类似，在使用队列前需要明确调用初始化过程
#dequeue_many(self,n,name)
queue_init = Queue.enqueue_many(([10,100],))

#FIFOQueue类的dequeue()函数可以将队列队首的一个元素出队
#dequeue(self,name)
a = Queue.dequeue()
b = a + 10
#FIFOQueue类的enqueue()函数可以将一个元素从队列队尾入队
#enqueue(self,vals,name)
Queue_en = Queue.enqueue([b])

with tf.Session() as sess:
    queue_init.run()
    for i in range(5):
        v, _ = sess.run([a, Queue_en])
        print(v)
        #打印输出队首的元素
        #10
        #100
        #20
        #110
        #30
