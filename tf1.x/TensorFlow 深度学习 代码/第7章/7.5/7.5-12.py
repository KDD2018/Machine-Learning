import matplotlib.pyplot as plt
import tensorflow as tf

image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.20, 0.3, 0.5, 0.5]]])

    #函数原型
    #sample_distorted_bounding_box(image_size,bounding_boxes,seed,seed2,min_object_covered,
    #       aspect_ratio_range,area_range,max_attempts,use_image_if_no_bounding_boxes,name)
    begin, size, bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(img_after_decode), bounding_boxes=boxes)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_after_decode, tf.float32), 0)
    image_boxed = tf.image.draw_bounding_boxes(batched, bounding_box)

    #slice()函数原型slice(input_,begin,size,name)
    sliced_image = tf.slice(img_after_decode,begin,size)

    plt.imshow(image_boxed[0].eval())
    plt.show()
    plt.imshow(sliced_image.eval())
    plt.show()
