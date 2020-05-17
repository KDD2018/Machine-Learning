import tensorflow as tf
import numpy as np
import collections
import random
import zipfile

# 出现频率最高的50000词作为单词表
vocabulary_size = 50000

file = "/home/jiangziyang/Word2vec/text8.zip"


def read_data(file):
    # ZipFile类的构造函数原型__init__(self,file,mode,compression,allowZip64)
    with zipfile.ZipFile(file=file) as f:
        # ZipFile类namelist()函数原型namelist(self)
        # ZipFile类read()函数原型read(self,name,pwd)
        original_data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return original_data


original_words = read_data(file)
# len()函数是Python中的内容，用于测试列表中元素的数量
print("len of original word is", len(original_words))
# 输出len of original words is 17005207


def build_vocabulary(original_words):
    # 创建一个名为count的列表，
    count = [["unkown", -1]]

    # Counter类构造函数原型__init__(args,kwds)
    # Counter类most_common()函数原型most_common(self,n)
    # extend()函数会在列表末尾一次性追加另一个序列中的多个值(用于扩展原来的列表）
    # 函数原型为extend(self,iterable)
    count.extend(collections.Counter(original_words).most_common(vocabulary_size - 1))

    # dict类构造函数原型__init__(self,iterable,kwargs)
    dictionary = dict()

    # 遍历count，并将count中按频率顺序排列好的单词装入dictionary，word为键
    # len(dictionary)为键值，这样可以在dictionary中按0到49999的编号引用单词
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()

    # unkown_count用于计数出现频率较低(属于未知)的单词
    unkown_count = 0

    # 遍历original_words原始单词列表，该列表并没有将单词按频率顺序排列好
    for word in original_words:

        if word in dictionary:  # original_words列表中的单词是否出现在dictionary中
            index = dictionary[word]  # 取得该单词在dictionary中的编号赋值给index
        else:
            index = 0  # 没有出现在dictionary中的单词，index将被赋值0
            unkown_count += 1  # 计数这些单词

        # 列表的append()方法用于扩充列表的大小并在列表的尾部插入一项
        # 如果用print(data)将data打印出来，会发现这里这里有很多0值
        # 使用print(len(data))会发现data长度和original_words长度相等，都是17005207
        data.append(index)

    # 将unkown类型的单词数量赋值到count列表的第[0][1]个元素
    count[0][1] = unkown_count

    # 反转dictionary中的键值和键，并存入另一个字典中
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# data, count, dictionary, reverse_dictionary = build_vocabulary(original_words)
# count[:5]是列表的切片操作，获取列表的前5个元素并作为一个新列表返回
# data[:10]在原理上是相同的
# 打印unkown类的词汇量及top4的单词的数量
# print("Most common words (+unkwon)", count[:5])
# 输出Most common words (+unkwon) [['unkown', 418391], ('the', 1061396),
#                          ('of', 593677), ('and', 416629), ('one', 411764)]
# 打印data中前十个单词及其编号
# print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])
# 输出Sample data [5235, 3084, 12, 6, 195, 2, 3137, 46, 59, 156]
# ['anarchism','originated','as','a','term','of','abuse','first','used','against']


data_index = 0
data, count, dictionary, reverse_dictionary = build_vocabulary(original_words)
def generate_batch(batch_size, num_of_samples, skip_distance):
    # 单词序号data_index定义为global变量，global是Python中的命名空间声明
    # 因为之后会多次调用data_index，并在函数内对其进行修改
    global data_index

    # 创建放置产生的batch和labels的容器
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    num_of_sample_words = 2 * skip_distance + 1

    #创建buffer队列，长度为num_of_sample_words，因为generate_batch()函数
    #会被调用多次，所以这里使用buffer队列暂存来自data的编号
    buffer = collections.deque(maxlen=num_of_sample_words)
    for _ in range(num_of_sample_words):
        buffer.append(data[data_index])
        data_index = (data_index + 1)


    # Python中//运算符会对商结果取整
    for i in range(batch_size // num_of_samples):
        #target=1，它在一个三元素列表中位于中间的位置，所以下标为skip_distance值
        #targets_to_avoid是生成样本时需要避免的单词列表
        target = skip_distance
        targets_to_avoid = [skip_distance]

        for j in range(num_of_samples):
            while target in targets_to_avoid:
                # 使用randint()函数用于产生0到num_of_sample_words-1之间的随机整数，
                # 使得target不在targets_to_avoid中
                target = random.randint(0, num_of_sample_words - 1)
            # 将需要避免的目标单词加入到列表targets_to_avoid中，在while后面使用append
            # 的方式可以避免target是两个重复的值，比如两个0
            targets_to_avoid.append(target)

            # i*num_skips+j最终会等于batch_size-1
            # 存入batch和labels的数据来源于buffer,而buffer中的数据来源于data
            # 也就是说，数组batch存储了目标单词在data中的索引
            # 而列表labels存储了语境单词(与目标单词相邻的单词)在data中的索引
            batch[i * num_of_samples + j] = buffer[skip_distance]
            labels[i * num_of_samples + j, 0] = buffer[target]

        # 在最外层的for循环使用append()函数将一个新的目标单词入队，清空队列最前面的单词
        buffer.append(data[data_index])
        data_index = (data_index + 1)
    return batch, labels
'''
batch, labels = generate_batch(batch_size=8, num_of_samples=2, skip_distance=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], 
          "->", labels[i, 0],reverse_dictionary[labels[i, 0]])
'''
'''打印的结果
    3082 originated -> 12 as
    3082 originated -> 5237 anarchism 
    12 as -> 3082 originated 
    12 as -> 6 a
    6 a -> 195 term
    6 a -> 12 as
    195 term -> 6 a
    195 term -> 2 of
    '''