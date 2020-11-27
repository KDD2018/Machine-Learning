from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import MobileNetV2
import tensorflow as tf
import json
import os
import PIL.Image as im
import numpy as np


def run():
    train_dir = '../flower_data/train'
    validation_dir = '../flower_data/validation'

    img_height = 224
    img_width = 224
    batch_size = 16
    epochs = 20


    def pre_function(img):
        # img = im.open('test.jpg')
        # img = np.array(img).astype(np.float32)
        img = img / 255.
        img = (img - 0.5) * 2.0
        return img


    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=pre_function)

    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(img_height, img_width),
                                                                  class_mode='categorical')
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n

    model = MobileNetV2(img_height, img_width, num_classes=5)

    # model.load_weights('pretrain_weights.ckpt')
    # for layer_t in model.layers[:-1]:
    #     layer_t.trainable = False
    model.summary()

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # not use softmax activition
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)


    @tf.function
    def test_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        test_loss(t_loss)
        test_accuracy(labels, output)


    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info

        # train
        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            rate = (step + 1) / (total_train // batch_size)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            acc = train_accuracy.result().numpy()
            print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
        print()

        # validate
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
            model.save_weights("../save_weights/MobileNetV2.ckpt", save_format="tf")


if __name__ == '__main__':
    run()