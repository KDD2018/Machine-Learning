import tensorflow as tf

# 省略了定义图上计算的过程，取而代之的是通过.meta文件直接加载持久化的图，
meta_graph = tf.train.import_meta_graph("/home/jiangziyang/model/model.ckpt.meta")

with tf.Session() as sess:
    # 使用restore()函数加载已经保存的模型
    meta_graph.restore(sess,"home/jiangziyang/model/model.ckpt")
    # 获取默认计算图上指定节点处的张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    #输出结果为Tensor(“add:0”, shape=(2,), dtype=float32)
    # import_meta_graph函数的原型是
    # import_meta_graph(meta_graph_or_file,clear_devics,import_scope,kwargs)
    # get_tensor_by_name()函数的原型是get_tensor_by_name(self,name)