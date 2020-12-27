from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import GoogLeNet
import tensorflow as tf
import json
import os
import numpy as np


train_dir = '../flower_data/train'
validation_dir = '../flower_data/validation'

# create direction for saving weights
if not os.path.exists("../save_weights"):
    os.makedirs("../save_weights")

img_height = 224
img_width = 224
batch_size = 16
epochs = 30


def pre_function(img: np.ndarray):
    img = img / 255.
    img = img - [0.485, 0.456, 0.406]
    img = img / [0.229, 0.224, 0.225]

    return img


def run():
    # 准备数据
    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode='categorical')
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(img_height, img_width),
                                                                  class_mode='categorical')

    total_train, total_val = train_data_gen.n, val_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 构建GoogLeNet模型
    model = GoogLeNet(img_height=img_height, img_width=img_width, class_num=5, aux_logits=True)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    # model.load_weights("pretrain_weights.ckpt")
    model.summary()

    # 使用keras低层API自定义训练过程
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            aux1, aux2, output = model(images, training=True)
            loss1 = loss_object(labels, aux1)
            loss2 = loss_object(labels, aux2)
            loss3 = loss_object(labels, output)
            loss = loss1*0.3 + loss2*0.3 + loss3
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def test_step(images, labels):
        _, _, output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss(t_loss)
        test_accuracy(labels, output)


    best_test_loss = float('inf')
    for epoch in range(1, epochs+1):
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info

        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images, labels)

        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            model.save_weights("../save_weights/GoogLeNet.h5")


if __name__ == '__main__':
    run()
