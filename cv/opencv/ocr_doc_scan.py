#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2 as cv
import numpy as np
import argparse
import pytesseract


parse = argparse.ArgumentParser()
parse.add_argument("-i", "--image", required=True, help="the path of image to be scanned")
args = parse.parse_args()
img_path = args.image


def show_cv(win_name, img):
    """
    显示图像
    :param win_name: 图像窗口名称
    :param img: 图像像素点值
    :return: None
    """
    cv.imshow(win_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    缩放图片
    :param image: 原图像素值
    :param width: 指定宽度
    :param height: 指定高度
    :param inter: 插值方法
    :return: 缩放后的图像像素值
    """
    h, w = image.shape[:2]
    if width:
        r = width / w
        dim = (width, int(r * h))
    elif height:
        r = height / h
        dim = (int(w * r), height)
    else:
        dim = None

    img_new = cv.resize(image, dsize=dim, interpolation=inter)
    return img_new


def order_points(pts):
    """
    确定四个点的先后顺序
    :param pts: 点
    :return:
    """
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype=np.float32)

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    进行透视变换
    :param image: 原图
    :param pts: 目标点
    :return: 透视变换后的图片
    """
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算最大宽度和最大高度
    width_b = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_t = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width_b), int(width_t))

    height_r = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_l = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(height_r), int(height_l))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, dsize=(maxWidth, maxHeight))

    return warped


if __name__ == '__main__':
    # 获取文档图片数据
    doc_img = cv.imread(img_path)
    print(doc_img.shape)
    show_cv('doc', doc_img)

    # 图片缩放
    img = resize(doc_img.copy(), height=500)
    show_cv('img', img)
    print(img.shape)

    # 预处理
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
    show_cv('G', img_gray)
    edge = cv.Canny(img_gray, threshold1=75, threshold2=200)
    show_cv('edge', edge)

    # 轮廓检测
    conts, hierarchy = cv.findContours(edge.copy(), mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    conts = sorted(conts, key=cv.contourArea, reverse=True)[:1]

    for c in conts:
        # 计算轮廓周长
        peri = cv.arcLength(curve=c, closed=True)
        # 做近似轮廓, c表示输入的点集, epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数, True表示封闭的
        approx = cv.approxPolyDP(curve=c, epsilon=0.02 * peri, closed=True)
        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            break
    cv.drawContours(img, contours=[screenCnt], contourIdx=-1, color=(0, 255, 0), thickness=2)
    show_cv('contours', img)

    # 将轮廓的四个点坐标对应到原图中去
    point_coord = screenCnt.reshape(4, 2) * doc_img.shape[0] / 500.0
    # 透视变换
    warped = four_point_transform(doc_img.copy(), point_coord)

    # 二值处理
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    ref = cv.threshold(warped, 100, 255, cv.THRESH_BINARY)[1]
    show_cv('ref', ref)
    # 图像旋转
    # rotated = np.rot90(ref)
    # show_cv('ro', rotated)
    cv.imwrite('scan.jpg', ref)

    # ocr
    text = pytesseract.image_to_string(ref)
    print(text)

