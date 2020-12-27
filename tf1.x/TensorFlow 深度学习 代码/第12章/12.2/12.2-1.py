import tensorflow as tf

# 构造函数NewCheckpointReader()能够读取checkpoint文件中最新保存的模型对应的
#　.index与.data文件，，该函数仅有filepattern一个参数，就是保存的模型的名称
reader = tf.train.NewCheckpointReader("/home/jiangziyang/model/model.ckpt")

#获取所有变量列表，得到的all_variables是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
# 打印的结果{'b': [2], 'a': [2]}

# 在一个for循环内遍历字典的内容
for variable_name in all_variables:
    print(variable_name, "shape is:",all_variables[variable_name])
    #输出 a shape is: [2]
    #     b shape is: [2]

# 使用get_tensor()函数获取张量的值，get_tensor()的函数原型为
# get_tensor(self, tensor_str)，其中tensor_str就是传入的张量字符串
print("Value for variable a is:",reader.get_tensor("a"))
#输出Value for variable a is: [1. 2.]
print("Value for variable b is:",reader.get_tensor("b"))
#输出Value for variable b is: [3. 4.]