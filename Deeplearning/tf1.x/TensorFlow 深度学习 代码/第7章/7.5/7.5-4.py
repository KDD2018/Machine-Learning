import matplotlib.pyplot as plt
import tensorflow as tf
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    #函数原型random_contrast(image,lower,upper,seed)
    #函数会在[lower upper]之间随机调整图像的对比度
    #但要注意参数lower和upper都不能为负
    adjusted_contrast = tf.image.random_contrast(img_after_decode, 0.2,18, )
    plt.imshow(adjusted_contrast.eval())
    plt.show()

    #adjust_contrast()函数原型
    #adjust_contrast(images,contrast_factor)
    #参数contrast_factor可为正或为负，正值会增加对比度，负值会降低对比度
    #在数值为整数时效果明显