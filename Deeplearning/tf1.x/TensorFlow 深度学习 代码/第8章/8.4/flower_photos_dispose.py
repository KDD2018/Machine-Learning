import glob
import os.path
import random
import numpy as np
from tensorflow.python.platform import gfile

input_data = "/home/jiangziyang/flower_photos"
CACHE_DIR = "/home/jiangziyang/datasets/bottleneck"


def create_image_dict():
    result = {}
    # path是flower_photos文件夹的路径，同时也包含了其子文件夹的路径
    # directory的数据形式为一个列表，打印其内容为：
    # /home/jiangziyang/flower_photos, /home/jiangziyang/flower_photos/daisy,
    # /home/jiangziyang/flower_photos/tulips, /home/jiangziyang/flower_photos/roses,
    # /home/jiangziyang/flower_photos/dandelion, /home/jiangziyang/flower_photos/sunflowers
    path_list = [x[0] for x in os.walk(input_data)]
    is_root_dir = True
    for sub_dirs in path_list:
        if is_root_dir:
            is_root_dir = False
            continue  # continue会跳出当前循环执行下一轮的循环

        # extension_name列表列出了图片文件可能的扩展名
        extension_name = ['jpg', 'jpeg', 'JPG', 'JPEG']
        # 创建保存图片文件名的列表
        images_list = []
        for extension in extension_name:
            # join()函数用于拼接路径，用extension_name列表中的元素作为后缀名，比如：
            # /home/jiangziyang/flower_photos/daisy/*.jpg
            # /home/jiangziyang/flower_photos/daisy/*.jpeg
            # /home/jiangziyang/flower_photos/daisy/*.JPG
            # /home/jiangziyang/flower_photos/daisy/*.JPEG
            file_glob = os.path.join(sub_dirs, '*.' + extension)

            # 使用glob()函数获取满足正则表达式的文件名，例如对于
            # /home/jiangziyang/flower_photos/daisy/*.jpg，glob()函数会得到该路径下
            # 所有后缀名为.jpg的文件，例如下面这个例子：
            # /home/jiangziyang/flower_photos/daisy/7924174040_444d5bbb8a.jpg
            images_list.extend(glob.glob(file_glob))

        # basename()函数会舍弃一个文件名中保存的路径，比如对于
        # /home/jiangziyang/flower_photos/daisy，其结果是仅仅保留daisy
        # flower_category就是图片的类别，这个类别通过子文件夹名获得
        dir_name = os.path.basename(sub_dirs)
        flower_category = dir_name

        # 初始化每个类别的flower photos对应的训练集图片名列表、测试集图片名列表
        # 和验证集图片名列表
        training_images = []
        testing_images = []
        validation_images = []

        for image_name in images_list:
            # 对于images_name列表中的图片文件名，它也包含了路径名，但我们不需要
            # 路径名所以这里使用basename()函数获取文件名
            image_name = os.path.basename(image_name)
            # random.randint()函数产生均匀分布的整数
            score = np.random.randint(100)
            if score < 10:
                validation_images.append(image_name)
            elif score < 20:
                testing_images.append(image_name)
            else:
                training_images.append(image_name)

        # 每执行一次最外层的循环，都会刷新一次result，result是一个字典，
        # 它的key为flower_category，它的value也是一个字典，以数据集分类的形式存储了
        # 所有图片的名称，最后函数将result返回
        result[flower_category] = {
            "dir": dir_name,
            "training": training_images,
            "testing": testing_images,
            "validation": validation_images,
        }
    return result


def get_image_path(image_lists, image_dir, flower_category, image_index, data_category):
    # category_list用列表的形式保存了某一类花的某一个数据集的内容，
    # 其中参数flower_category从函数get_random_bottlenecks()传递过来
    category_list = image_lists[flower_category][data_category]

    # actual_index是一个图片在category_list列表中的位置序号
    # 其中参数image_index也是从函数get_random_bottlenecks()传递过来
    actual_index = image_index % len(category_list)

    # image_name就是图片的文件名
    image_name = category_list[actual_index]

    # sub_dir得到flower_photos中某一类花所在的子文件夹名
    sub_dir = image_lists[flower_category]["dir"]

    # 拼接路径，这个路径包含了文件名，最终返回给create_bottleneck()函数
    # 作为每一个图片对应的特征向量的文件
    full_path = os.path.join(image_dir, sub_dir, image_name)
    return full_path


def create_bottleneck(sess, image_lists, flower_category, image_index,
                      data_category, jpeg_data_tensor, bottleneck_tensor):
    # sub_dir得到的是flower_photos下某一类花的文件夹名，这类花由
    # flower_photos参数确定，花的文件夹名由dir参数确定
    sub_dir = image_lists[flower_category]["dir"]

    # 拼接路径，路径名就是在CACHE_DIR路径的基础上加上sub_dir
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)

    # 判断拼接出的路径是否存在，如果不存在，则在CACHE_DIR下创建相应的子文件夹
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    # 获取一张图片对应的特征向量的全名，这个全名包括了路径名，而且会在图片的.jpg后面
    # 用.txt作为后缀，获取没有.txt缀的文件名使用了get_image_path()函数，
    # 该函数会返回带路径的图片名
    bottleneck_path = get_image_path(image_lists, CACHE_DIR, flower_category,
                                     image_index, data_category) + ".txt"

    # 如果指定名称的特征向量文件不存在，则通过InceptionV3模型计算得到该特征向量
    # 计算的结果也会存入文件
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片名，这个图片名包含了原始图片的完整路径
        image_path = get_image_path(image_lists, input_data, flower_category,
                                    image_index, data_category)
        # 读取图片的内容
        image_data = gfile.FastGFile(image_path, "rb").read()

        # 将当前图片输入到InceptionV3模型，并计算瓶颈张量的值，所得瓶颈张量的值
        # 就是这张图片的特征向量，但是得到的特征向量是四维的，所以还需要通过squeeze()
        # 函数压缩成一维的，以方便作为全连层的输入
        bottleneck_values = sess.run(bottleneck_tensor, feed_dict={jpeg_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)

        # 将计算得到的特征向量存入文件，存入文件前需要为两个值之间加入逗号作为分隔
        # 这样可以方便从文件读取数据时的解析过程
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, "w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # else是特征向量文件已经存在的情况，此时会直接从bottleneck_path获取
        # 特征向量数据
        with open(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        # 从文件读取的特征向量数据是字符串的形式，要以逗号为分隔将其转为列表的形式
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_bottlenecks(sess, num_classes, image_lists, batch_size, data_category, jpeg_data_tensor,
                           bottleneck_tensor):
    # 定义bottlenecks用于存储得到的一个batch的特征向量
    # 定义labels用于存储这个batch的label标签
    bottlenecks = []
    labels = []

    for i in range(batch_size):
        # random_index是从五个花类中随机抽取的类别编号
        # image_lists.keys()的值就是五种花的类别名称
        random_index = random.randrange(num_classes)
        flower_category = list(image_lists.keys())[random_index]

        # image_index就是随机抽取的图片的编号，在get_image_path()函数中
        # 我们会看到如何通过这个图片编号和random_index确定类别找到图片的文件名
        image_index = random.randrange(65536)

        # 调用get_or_create_bottleneck()函数获取或者创建图片的特征向量
        # 这个函数调用了get_image_path()函数
        bottleneck = create_bottleneck(sess, image_lists, flower_category, image_index,
                                       data_category, jpeg_data_tensor, bottleneck_tensor)

        # 首先生成每一个标签的答案值，再通过append()函数组织成一个batch列表
        # 函数将完整的列表返回
        label = np.zeros(num_classes, dtype=np.float32)
        label[random_index] = 1.0
        labels.append(label)
        bottlenecks.append(bottleneck)
    return bottlenecks, labels



def get_test_bottlenecks(sess, image_lists, num_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    labels = []

    #flower_category_list是image_lists中键的列表，打印出来就是这样：
    #['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']
    flower_category_list = list(image_lists.keys())

    data_category = "testing"

    #枚举所有的类别和每个类别中的测试图片
    #在外层的for循环中，label_index是flower_category_list列表中的元素下标
    #flower_category就是该列表中的值
    for label_index, flower_category in enumerate(flower_category_list):

        #在内层的for循环中，通过flower_category和"testing"枚举image_lists中每一类花中
        #用于测试的花名，得到的名字就是unused_base_name，但我们只需要image_index
        for image_index, unused_base_name in enumerate(image_lists[flower_category]
                                                                        ["testing"]):

            #调用create_bottleneck()函数创建特征向量，因为在进行训练或验证的过程中
            #用于测试的图片并没有生成相应的特征向量，所以这里要一次性全部生成
            bottleneck = create_bottleneck(sess, image_lists, flower_category,
                                                    image_index,data_category,
                                          jpeg_data_tensor, bottleneck_tensor)

            #接下来就和get_random_bottlenecks()函数相同了
            label = np.zeros(num_classes, dtype=np.float32)
            label[label_index] = 1.0
            labels.append(label)
            bottlenecks.append(bottleneck)
    return bottlenecks, labels