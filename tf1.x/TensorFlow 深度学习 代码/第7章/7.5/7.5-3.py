import matplotlib.pyplot as plt
import tensorflow as tf
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    #函数原型random_brightness(image,max_delta,seed)
    #max_delta的值不能为负，函数会在[-max_delta，max_delta]值之间随机调整图像的亮度
    adjusted_brightness = tf.image.random_brightness(img_after_decode,max_delta=1)
    plt.imshow(adjusted_brightness.eval())
    plt.show()

    #adjust_brightness()函数原型
    #adjust_brightness(image,delta)
    #说明：delta参数为正值则图像的亮度会增加，为负值则图像的亮度会降低
    #将dalta设为大于1或小于-1的值是没有意义的，这时图像会变成一致的颜色