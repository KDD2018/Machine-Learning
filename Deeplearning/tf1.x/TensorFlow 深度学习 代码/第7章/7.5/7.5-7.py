import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    #函数原型resize_images(images,size,method,align_corners)
    resized = tf.image.resize_images(img_after_decode, [300, 300], method=3)

    print(resized.dtype)
    #打印的信息<dtype: 'uint8'>

    # 从print的结果看出经由resize_images()函数处理图片后返回的数据是float32格式的，
    # 所以需要转换成uint8才能正确打印图片，这里使用np.asarray()存储了转换的结果
    resized = np.asarray(resized.eval(), dtype="uint8")

    plt.imshow(resized)
    plt.show()