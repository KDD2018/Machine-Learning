from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
from tensorflow import keras
from Parking import Parking
import pickle

cwd = os.getcwd()


def img_process(test_images, park):
    """
    图像处理
    :param test_images: 待处理图像
    :param park: 处理方法类实例
    :return:
    """
    # 过滤
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)
    # 转灰度图
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)
    # 边缘检测
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)
    # 选出停车位区域范围
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # 画出停车位的轮廓直线
    list_of_lines = list(map(park.hough_lines, roi_images))

    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines))
    park.show_images(line_images)

    # 画出每一停车位的轮廓
    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)

    park.show_images(rect_images)

    # 画出每排停车位的矩形轮廓
    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    park.show_images(delineated)
    final_spot_dict = spot_pos[1]
    print(len(final_spot_dict))

    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict

def keras_model(weights_path):
    """
    加载模型
    :param weights_path: 模型路径
    :return:
    """
    model = keras.models.load_model(weights_path)
    return model

def img_test(test_images, final_spot_dict, model, class_dictionary):
    """
    测试模型
    :param test_images: 测试集
    :param final_spot_dict:停车位字典
    :param model: 模型
    :param class_dictionary: 类别字典
    :return:
    """
    for i in range(len(test_images)):
        predicted_images = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)


def video_test(video_name, final_spot_dict, model, class_dictionary):
    name = video_name
    cap = cv2.VideoCapture(name)
    park.predict_on_video(name, final_spot_dict, model, class_dictionary, ret=True)


if __name__ == '__main__':

    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    weights_path = 'car1.h5'
    video_name = 'parking_video.mp4'
    class_dictionary = {0: 'empty', 1: 'occupied'}
    park = Parking()
    park.show_images(test_images)
    final_spot_dict = img_process(test_images, park)
    model = keras_model(weights_path)
    img_test(test_images, final_spot_dict, model, class_dictionary)
    # video_test(video_name, final_spot_dict, model, class_dictionary)
