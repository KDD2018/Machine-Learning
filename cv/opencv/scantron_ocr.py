#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def order_points(pts):
    """
    确定四点位置顺序
    :param pts: 四点坐标
    :return: 顺序四点坐标
    """
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

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


def show_cv(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    """
    对轮廓进行排序
    :param cnts: 轮廓坐标
    :param method: 排序方法
    :return:
    """
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


if __name__ == '__main__':
    # 获取答题卡图片
    scantron = cv.imread(args['image'])
    scan_copy = scantron.copy()
    show_cv('original image', scantron)
    # 灰度图
    gray = cv.cvtColor(scantron, cv.COLOR_BGR2GRAY)
    show_cv('gray', gray)
    # 高斯滤波平滑
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    show_cv('blur', blurred)
    # 边缘检测
    edge1 = cv.Canny(blurred, 75, 200)
    # edge2 = cv.Canny(blurred, 150, 200)
    # res = np.hstack((edge1, edge2))
    # show_cv('canny', res)
    # 轮廓检测
    contours, hierarchy = cv.findContours(edge1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(scan_copy, contours, -1, (0, 0, 255), 3)
    show_cv('contours', scan_copy)

    tarcnt = None
    if len(contours) > 0:
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        for cnt in contours:
            # 近似轮廓
            approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
            if len(approx) == 4:
                tarcnt = approx
                break
    # 透视变换
    warped = four_point_transform(gray, tarcnt.reshape(4, 2))
    show_cv('warped', warped)
    # Otsu's 阈值处理
    ret, thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    show_cv('thresh', thresh)
    thresh_cp = thresh.copy()
    # 找到每一个圆圈轮廓
    cnts, hierarchy = cv.findContours(thresh_cp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(thresh_cp, cnts, -1, (255, 255, 255), 3)
    show_cv('thresh_Contours', thresh_cp)

    # 滤出答案轮廓
    questionCnts = []
    for c in cnts:
        # 本案例中的答案选项在圆形内，其外接矩形是正方形
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    # 按照从上到下进行排序
    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # 排序
        cnts = sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        # 遍历每一个结果
        for (j, c) in enumerate(cnts):
            # 使用mask来判断结果
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)  # -1表示填充
            show_cv('mask', mask)
            # 通过计算非零点数量来算是否选择这个答案
            mask = cv.bitwise_and(thresh, thresh, mask=mask)
            total = cv.countNonZero(mask)

            # 通过阈值判断
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # 对比正确答案
        color = (0, 0, 255)
        k = ANSWER_KEY[q]

        # 判断正确
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        # 绘图
        cv.drawContours(warped, [cnts[k]], -1, color, 3)

    score = (correct / 5.0) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv.putText(warped, "{:.2f}%".format(score), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv.imshow("Original", scantron)
    cv.imshow("Exam", warped)
    cv.waitKey(0)