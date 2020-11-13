import os
import random
from shutil import copy


def mk_file(file):
    """
    创建文件夹
    :param file:
    :return:
    """
    if not os.path.exists(file):
        os.makedirs(file)


def split_data(raw_path, class_names, split_rate=0.1):
    """

    :param raw_path:
    :param class_names:
    :param split_rate:
    :return:
    """
    for cla in class_names:
        cla_path = raw_path + '/' + cla + '/'
        images = os.listdir(cla_path)
        num = len(images)
        valid_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in valid_index:
                image_path = cla_path + image
                new_path = './flower_data/validation/' + cla
            else:
                image_path = cla_path + image
                new_path = './flower_data/train/' + cla
            copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()


# 源数据路径
data_root = os.path.abspath(os.path.dirname(os.getcwd()))
image_path = data_root + '/flower_photos'

# 标签类别
class_names = [item for item in os.listdir(image_path) if '.txt' not in item]

data_dir = './flower_data'


if __name__ == '__main__':
    # 创建训练集和验证集对应的文件夹
    for item in class_names:
        mk_file(os.path.join('flower_data/train', item))
        mk_file(os.path.join('flower_data/validation', item))
    # 拆分数据
    split_data(raw_path=image_path, class_names=class_names)