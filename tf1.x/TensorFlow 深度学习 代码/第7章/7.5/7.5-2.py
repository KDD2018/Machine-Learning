import matplotlib.pyplot as plt
import tensorflow as tf
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()

with tf.Session() as sess:
    #decode
    img_after_decode = tf.image.decode_png(image)
    # 以一定概率左右翻转图片。
    # 函数原型为random_flip_left_right(image,seed)
    flipped = tf.image.random_flip_left_right(img_after_decode)

    # 用pyplot工具显示
    plt.imshow(flipped.eval())
    plt.show()

    #以一定概率上下翻转图片。
    #image.random_flip_up_down(image,seed)
    #将图像进行上下翻转
    #image.flip_up_down(image)
    #将图像进行左右翻转
    #lipped2 = tf.image.flip_left_right(image)
    #将图像进行对角线翻转
    #image.transpose_image(image)
