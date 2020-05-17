# 导入os库是因为需要拼接路径
import os
import tensorflow as tf

num_classes = 10

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10数据
class CIFAR10Record(object):
    pass


# 定义读取Cifar-10数据的函数
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1  # 如果是Cifar-100数据集，则此处为2
    result.height = 32
    result.width = 32
    result.depth = 3  # 因为是RGB三通道，所以深度为3

    image_bytes = result.height * result.width * result.depth  # =3072

    # 每个样本都包含一个label数据和image数据，结果为record_bytes=3073
    record_bytes = label_bytes + image_bytes

    # 创建一个文件读取类，并调用该类的read()函数从文件队列中读取文件
    # FixedLengthRecordReader类用于读取固定长度字节数信息(针对bin文件而言，使用
    # FixedLengthRecordReader读取比较合适)，在11.1节介绍文件读取的时候会介绍与之
    # 相似的TFRecordReader类，如果想了解更多信息，可以快速翻阅第十一章
    # 构造函数原型__init__(self,record_bytes,header_bytes,footer_bytes,name)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    # 得到的value就是record_bytes长度的包含一个label数据和image数据字符串
    # decode_raw()函数可以将字符串解析成图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 将得到的record_bytes数组中的第一个元素类型转换为int32类型
    # strided_slice()函数用于对input截取从[begin, end)区间的数据
    # 函数原型strided_slice(input,begin,end,strides,begin_mask,end_mask,
    #                            ellipsis_mask,new_axis_mask,shrink_axis_mask,name)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    # 剪切label之后剩下的就是图片数据,我们将这些数据的格式从[depth * height * width]
    # 转为[depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    # 将[depth, height, width]的格式转变为[height, width, depth]的格式
    # transpose()函数用于原型为     transpose(x,perm,name)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


# inputs()函数调用了read_cifar10()函数，可以选择是否对读入的数据进行数据增强处理
def inputs(data_dir, batch_size, distorted):
    # 使用os的join()函数拼接路径
    filenames = [os.path.join(data_dir,"data_batch_%d.bin" % i) for i in range(1, 6)]

    # 创建一个文件队列，并调用read_cifar10()函数读取队列中的文件
    # 关于队列的内容可快速翻阅第十一章
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)

    # 使用cast()函数将图片数据转为float32格式，原型cast(x,DstT,name)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    num_examples_per_epoch = num_examples_pre_epoch_for_train

    # 对图像数据进行数据增强处理
    if distorted != None:
        # 将[32,32,3]大小的图片随机裁剪成[24,24,3]大小
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])

        # 随机左右翻转图片
        flipped_image = tf.image.random_flip_left_right(cropped_image)

        # 使用random_brightness()函数调整亮度
        # 函数原型random_brightness(image,max_delta,seed)
        adjusted_brightness = tf.image.random_brightness(flipped_image,
                                                         max_delta=0.8)

        # 调整对比度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness,
                                                     lower=0.2, upper=1.8)

        # 标准化图片，注意不是归一化
        #per_image_standardization()是对每一个像素减去平均值并处以像素方差
        #函数原型per_image_standardization(image)
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图片数据及label的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train. '
                             'This will take a few minutes.' % min_queue_examples)

        #使用shuffle_batch()函数随机产生一个batch的image和label
        #函数原型shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue,
        #      num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)
        images_train, labels_train = tf.train.shuffle_batch(
                                            [float_image, read_input.label],
                                          batch_size=batch_size, num_threads=16,
                                        capacity=min_queue_examples + 3 * batch_size,
                                                min_after_dequeue=min_queue_examples)
        return images_train, tf.reshape(labels_train, [batch_size])

    # 不对图像数据进行数据增强处理
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)

        #没有图像的其他处理过程，直接标准化
        float_image = tf.image.per_image_standardization(resized_image)

        #设置图片数据及label的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch *0.4)

        # 使用batch()函数创建样例的batch，这个过程使用最多的是shuffle_batch()函数
        # 但是这里使用batch()函数代替了shuffle_batch()函数
        #batch()函数原型batch(tensor_list, batch_size, num_threads=1, capacity=32,
        #                              enqueue_many=False, shapes=None, name=None)
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                     batch_size=batch_size,num_threads=16,
                                     capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])