import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型expand_dims(input,axis,name,dim)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_after_decode, tf.float32), 0)

    # 定义边框的坐标系数
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.20, 0.3, 0.5, 0.5]]])

    # 绘制边框，函数原型draw_bounding_boxes(images,boxes,name)
    image_boxed = tf.image.draw_bounding_boxes(batched, boxes)

    # draw_bounding_boxes()函数处理的是一个batch的图片，如果此处给imshow()函数
    # 传入image_boxed参数会造成报错(Invalid dimensions for image data)
    plt.imshow(image_boxed[0].eval())
    plt.show()
