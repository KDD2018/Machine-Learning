#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras


img_width, img_height = 48, 48
data_dir = "train_data"
num_class = 2
batch_size = 32
epochs = 50

def get_images_num(data_dir):
    """
    获取训练集
    :param path:
    :return:
    """
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'test')
    train_occupied_dir = os.path.join(train_dir, 'occupied')
    train_empty_dir = os.path.join(train_dir, 'empty')
    validation_occupied_dir = os.path.join(validation_dir, 'occupied')
    validation_empty_dir = os.path.join(validation_dir, 'empty')
    total_train = len(os.listdir(train_empty_dir)) + len(os.listdir(train_occupied_dir))
    total_val = len(os.listdir(validation_occupied_dir)) + len(os.listdir(validation_empty_dir))
    return total_train, total_val, train_dir, validation_dir


def image_generator(data_dir):
    """
    图像生成器
    :param data_dir: 数据路径
    :return: 生成器
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                           horizontal_flip=True,
                                                           fill_mode="nearest",
                                                           zoom_range=0.1,
                                                           width_shift_range=0.1,
                                                           height_shift_range=0.1,
                                                           rotation_range=5)
    generator = datagen.flow_from_directory(data_dir,
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size,
                                            class_mode="categorical")
    return generator


def creat_model():
    """
    创建编译模型
    :return: model
    """
    # input_ = tf.keras.Input(shape=(img_width, img_height, 3))
    model_vgg16 = keras.applications.VGG16(weights='imagenet',
                                           include_top=False,
                                           input_shape=(img_width, img_height, 3))
    for layer in model_vgg16.layers[:10]:
        layer.trainable = False
    x = keras.layers.Flatten()(model_vgg16.output)
    prediction = keras.layers.Dense(num_class, activation='softmax')(x)

    myModel = tf.keras.Model(inputs=model_vgg16.input, outputs=prediction)
    myModel.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
                    metrics=['accuracy'])
    return myModel


if __name__ == '__main__':
    # 获悉训练集、验证集样本数、路径
    total_train, total_val, train_dir, validation_dir = get_images_num(data_dir)
    # 训练集、验证集数据生成器
    train_generator = image_generator(train_dir)
    validation_generator = image_generator(validation_dir)
    # 创建编译模型
    my_model = creat_model()
    # 训练模型
    checkpoint = keras.callbacks.ModelCheckpoint("car1.h5",
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='max', period=1)
    early = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
    history_object = my_model.fit_generator(train_generator,
                                            steps_per_epoch=total_train // batch_size,
                                            epochs=epochs,
                                            validation_data=validation_generator,
                                            validation_steps=total_val // batch_size,
                                            callbacks=[checkpoint, early])


