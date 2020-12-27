import tensorflow as tf
# gfile模块定义在tensorflow/python/platform/gfile.py
# 包含GFile、FastGFile和Open三个没有线程锁定的文件I/O包装器类
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    # 使用FsatGFile类的构造函数返回一个FastGFile类
    with gfile.FastGFile("/home/jiangziyang/model/model.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        # 使用FastGFile类的read()函数读取保存的模型文件，并以字符串形式
        # 返回文件的内容，之后通过ParseFromString()函数解析文件的内容
        graph_def.ParseFromString(f.read())

    # 使用import_graph_def()函数将graph_def中保存的计算图加载到当前图中
    # 原型import_graph_def(graph_def,input_map,return_elements,name,op_dict,
    #                                                     producer_op_list)
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])

    print(sess.run(result))
    # 输出为[array([3.], dtype=float32)]