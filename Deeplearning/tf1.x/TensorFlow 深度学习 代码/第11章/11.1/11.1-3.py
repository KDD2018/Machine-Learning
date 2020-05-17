import  tensorflow as tf

#创建一个队列对输入文件列表进行维护，队列的知识放到了本章的稍后
filename_queue = tf.train.string_input_producer(["/home/jiangziyang/CSV/data.csv"])

#调用构造函数实例化一个TextLineReader类
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#是用于指定矩阵格式以及数据类型的，CSV文件中的矩阵，是mxn的，则此处为1xn，
#[1]中的1用于指定数据类型，比如矩阵中如果有小数，则为float,[1]应该变为[1.0]。
record_defaults = [[1.0], [1.0], [1.0], [1.0],[1.0]]

#col是矩阵的列数，一般会多写一列
#decode_csv()函数原型decode_csv(records,record_defaults,field_delim,name)
col1, col2, col3, col4,col5 = tf.decode_csv(value, record_defaults=record_defaults)

#concat()函数用于拼接要求参与拼接的数据拥有一致的数据类型
#concat()函数原型为concat(values,axis,name)
#axis参数选择拼接的维度，axis=0则在某一个shape的第一个维度上(行)进行拼接，
#axis=1则在某一个shape的第二个维度上(列)进行拼接
features = tf.concat([[col1], [col2], [col3]],0)


with tf.Session() as sess:
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coordinator)
    for i in range(30):
        example, label = sess.run([features, col4])
        print(example)
        print(label)

    coordinator.request_stop()
    coordinator.join(threads)