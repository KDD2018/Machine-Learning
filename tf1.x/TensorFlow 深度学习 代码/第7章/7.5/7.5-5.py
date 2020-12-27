import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型adjust_hue(image,delta,name)
    adjusted_hue = tf.image.adjust_hue(img_after_decode, 0.1)
    adjusted_hue = tf.image.adjust_hue(img_after_decode, 0.3)
    adjusted_hue = tf.image.adjust_hue(img_after_decode, 0.6)
    adjusted_hue = tf.image.adjust_hue(img_after_decode, 0.9)

    plt.imshow(adjusted_hue.eval())
    plt.show()

    # random_hue()函数原型为random_hue(image, max_delta)
    # 功能是在[-max_delta, max_delta]的范围随机调整图片的色相。
    # max_delta的取值在[0, 0.5]之间。