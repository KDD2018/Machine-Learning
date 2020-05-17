import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型central_crop(image,central_fraction)
    central_cropped = tf.image.central_crop(img_after_decode, 0.4)
    plt.imshow(central_cropped.eval())
    plt.show()