#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2 as cv
import argparse
import numpy as np
from imutils import contours as ct


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-t', '--template', required=True, help='path to template OCR-A image')
args = vars(ap.parse_args())


# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


def sort_contours(cnts, method="left-to-right"):
    """
    轮廓排序
    :param cnts: 轮廓列表
    :param method: 排序方法
    :return: 排序后的轮廓
    """
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # 获取每个数字的外接矩形——x,y,w,h
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    # 将每个数字轮廓与外接矩形组成元组，并按照外接矩形的坐标x或y进行排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    自定义图片大小
    :param image: 原图
    :param width: 设定宽度
    :param height: 设定高度
    :param inter: 插值方法
    :return: 自定义大小的图片
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


def show_cv(name, img):
    """
    opencv绘制图像
    :param name: 图片窗口名称
    :param img: 图片路径
    :return: None
    """
    cv.imshow(winname=name, mat=img)
    cv.waitKey(delay=0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # 读取模板转化为二值图像
    temp = cv.imread(args['template'])
    show_cv('template', temp)
    print('原始模板图像形状: ', temp.shape)
    temp_gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    show_cv('gray template', temp_gray)
    print('\n模板的灰度图像形状: ', temp_gray.shape)
    # 大于thresh的像素点设为0， 否则设为maxval
    temp_bin = cv.threshold(temp_gray, thresh=10, maxval=255, type=cv.THRESH_BINARY_INV)[1]
    show_cv('binary template', temp_bin)
    print('\n模板的二值化图像形状: ', temp_bin.shape)

    # 检测模板轮廓

    # cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    # 返回的contours存放每个轮廓，hierarchy存放的是轮廓的等级、从属关系
    contours, hierarchy = cv.findContours(image=temp_bin.copy(),
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image=temp, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
    show_cv('temp with contours', temp)
    print('\n轮廓个数: ', len(contours))

    # 分开保存模板中所有轮廓
    contours_reverse = sort_contours(contours, method="left-to-right")[0]
    digits = {}
    for i, cont in enumerate(contours_reverse):
        # 计算外接矩形并保存程统一大小
        (x, y, w, h) = cv.boundingRect(cont)
        roi = temp_bin[y: y + h, x: x + w]
        roi = cv.resize(src=roi, dsize=(57, 88))
        # show_cv(str(i), roi)
        digits[i] = roi


    # 加载处理待检图像
    img = cv.imread(args['image'])
    print('\n待检图像形状: ', img.shape)
    show_cv('img', img)
    img = resize(img, width=300)
    # 转为灰度图
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    show_cv('gray img', gray_img)
    print('\n自定义大小为: ', gray_img.shape)

    # 初始化卷积核
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    # 礼帽操作，突出更明亮的区域
    tophat = cv.morphologyEx(gray_img, cv.MORPH_TOPHAT, rectKernel)
    show_cv('tophat', tophat)
    # 边缘检测
    gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1) # ksize=-1相当于用3*3的
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    # print(np.array(gradX).shape)
    show_cv('gradX', gradX)

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
    show_cv('gradX', gradX)
    # THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
    thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    show_cv('thresh', thresh)
    # 再来一个闭操作
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)
    show_cv('thresh', thresh)

    # 计算轮廓
    threshCnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img.copy(), threshCnts, -1, (0, 0, 255), 3)
    show_cv('img', img)
    locs = []
    # 遍历轮廓, 将ID行的轮廓保存下来
    for (i, c) in enumerate(threshCnts):
        # 计算矩形
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)
        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4.0:
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # 符合的留下来
                locs.append((x, y, w, h))
    # 将符合的轮廓从左到右排序
    locs = sorted(locs, key=lambda x: x[0])


    output = []
    # 遍历每一个轮廓中的数字
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # 根据坐标提取每一个组
        group = gray_img[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        show_cv('group', group)
        # 预处理
        group = cv.threshold(group, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        show_cv('group', group)
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        digitCnts = ct.sort_contours(digitCnts, method="left-to-right")[0]
        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = cv.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv.resize(roi, (57, 88))
            show_cv('roi', roi)

            # 计算匹配得分
            scores = []
            # 在模板中计算每一个得分
            for (digit, digitROI) in digits.items():
                # 模板匹配
                result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)
                (_, score, _, _) = cv.minMaxLoc(result)
                scores.append(score)
            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
        # 画出来
        cv.rectangle(img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv.putText(img, "".join(groupOutput), (gX, gY - 15), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # 得到结果
        output.extend(groupOutput)

    # 打印结果
    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))
    cv.imshow("Image", img)
    cv.waitKey(0)