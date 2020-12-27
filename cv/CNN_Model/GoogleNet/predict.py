from model import GoogLeNet
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt


def predict(img_path, weights_path, class_indices_path):
    """
    对图像进行预测分类
    :param img_path: 待预测图像路径
    :param weights_path: 模型权重路径
    :param class_indices_path: 标签类别索引
    :return: 待预测图像类别
    """
    img_height = img_width = 224

    # 加载待预测图像
    img = Image.open(img_path)
    # 重设图像大小
    img = img.resize((img_width, img_height))
    plt.imshow(img)

    # 归一化
    img = (np.array(img) / 255. - 0.5) / 0.5

    # 增加batch这个维度
    img = (np.expand_dims(img, 0))

    # 加载标签类别索引文件
    try:
        json_file = open(class_indices_path, 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # 预测
    model = GoogLeNet(class_num=5, aux_logits=False)
    # 由于当前创建的模型没有辅助分类器，而保存的训练模型有辅助分类器，需要by_name参数根据名字匹配相关权重参数
    model.load_weights(weights_path, by_name=True)
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    label = class_indict[str(predict_class)], result[predict_class]
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    img_path = '../flower_photos/roses/12240303_80d87f77a3_n.jpg'
    weights_path = '../save_weights/GoogLeNet.h5'
    class_indices_path = '../class_indices.json'
    predict(img_path, weights_path, class_indices_path)