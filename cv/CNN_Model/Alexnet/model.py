import tensorflow as tf
from tensorflow import keras
import numpy as np


def AlexNet_v1(img_height=224, img_width=224, class_num=1000):
    """
    Build the Model by Functional API
    :param img_height: the height of input image
    :param img_width: the width of input image
    :param class_num: optional number of classes to classify images into
    :return: A 'keras.Model' instance
    """
    img_input = keras.layers.Input(shape=(img_height, img_width, 3), dtype='float32')
    x = keras.layers.ZeroPadding2D(((1, 2), (1, 2)))(img_input)
    x = keras.layers.Conv2D(48, kernel_szie=11, strides=4, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = keras.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dense(class_num)(x)
    outputs = keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=img_input, outputs=outputs)
    return model


class AlexNet_v2(tf.keras.Model):
    """Build the Model by Subclassing API"""
    def __init__(self, class_num=1000):
        super(AlexNet_v2, self).__init__()
        self.features = tf.keras.Sequential([
            keras.layers.ZeroPadding2D(((1, 2), (1, 2))),
            keras.layers.Conv2D(48, kernel_size=11, strides=4, activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2),
            keras.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2),
            keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2)
        ])

        self.flatten = keras.layers.Flatten()
        self.classifier = tf.keras.Sequential([
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dense(class_num),
            keras.layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

