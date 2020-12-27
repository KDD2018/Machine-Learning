import matplotlib.pyplot as plt
import tensorflow as tf
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    #函数原型resize_image_with_crop_or_pad(image,target_height,target_width)
    #裁剪图像
    croped = tf.image.resize_image_with_crop_or_pad(img_after_decode, 300, 300)
    #填充图像
    padded = tf.image.resize_image_with_crop_or_pad(img_after_decode, 1000, 1000)

    #用pyplot显示结果
    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()