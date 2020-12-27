import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型
    # crop_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    # pad_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    croped = tf.image.crop_to_bounding_box(img_after_decode, 100, 100, 300, 300)
    padded = tf.image.pad_to_bounding_box(img_after_decode, 100, 100, 1000, 1000)

    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()
