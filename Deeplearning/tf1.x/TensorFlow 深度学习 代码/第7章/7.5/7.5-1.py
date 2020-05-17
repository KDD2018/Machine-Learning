import matplotlib.pyplot as plt
import tensorflow as tf

# 读取原始的图像
image = tf.gfile.FastGFile("/home/jiangziyang/images/duck.png", 'r').read()
with tf.Session() as sess:
    # TensorFlow提供了decode_png()函数将.png格式的图像解码从而得到图像对应的三位矩阵
    # 函数原型decode_png(contents,channels,dtype,name)
    img_after_decode = tf.image.decode_png(image)

    # 输出解码之后的三维矩阵，并调用pyplot工具可视化得到的图像
    print(img_after_decode.eval())
    plt.imshow(img_after_decode.eval())
    plt.show()

    # 这一句是为了方便后续的样例程序对图像进行处理
    # img_after_decode = tf.image.convert_image_dtype(img_after_decode,dtype = tf.float32)

    # TensorFlow提供了encode_png()函数将解码后的图像进行再编码
    # 函数原型encode_png(image,compression,name)
    encode_image = tf.image.encode_png(img_after_decode)
    with tf.gfile.GFile("/home/jiangziyang/images2/duck.png", "wb") as f:
        f.write(encode_image.eval())

    '''
    [[[56  55  60 255]
      [56  55  60 255]
      [56  55  60 255]
      ...
      [56  55  60 255]
      [56  55  60 255]
      [56  55  60 255]]

     [[ 57  45  38 255]
      [ 58  46  38 255]
      [ 58  46  38 255]
      ...
      [ 30  21  18 255]
      [ 32  23  19 255]
      [ 32  23  20 255]]

    [[ 70  55  47 255]
     [ 71  55  48 255]
     [ 71  56  49 255]
     ...
     [ 40  29  24 255]
     [ 41  30  26 255]
     [ 42  32  27 255]]
     ...

    [[  9  32  57 255]
     [  9  34  57 255]
     [  5  30  55 255]
     ...
     [145 178 231 255]
     [146 174 229 255]
     [149 178 233 255]]

    [[  7  42  79 255]
     [  3  47  81 255]
     [  3  51  88 255]
     ...
     [123 164 223 255]
     [124 157 219 255]
     [137 170 233 255]]

    [[  6  35  75 255]
     [  2  39  85 255]
     [  1  49 104 255]
     ...
     [ 93 145 213 255]
     [101 151 220 255]
     [ 88 142 217 255]]]
   '''
    # decode_jpeg()函数用于解码.jpeg/.jpg格式的图像，原型
    # decode_jpeg(contents,channels,ratio,fancy_upscaling,try_recover_truncated,
    #                                           acceptable_fraction,dct_method,name)
    # decode_gif()函数用于解码.gif格式的图像，原型
    # decode_gif(contents,name)

    # encode_jpeg()函数用于编码为.jpeg/.jpg格式的图像，原型
    # encode_jpeg(image,format,quality,progressive,optimize_size,chroma_downsampling,
    #                                density_unit,x_density,y_density,xmp_metadata,name)

