import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import AlexNet_v1, AlexNet_v2
import matplotlib.pyplot as plt
import json
import os


data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
image_path = data_root + '/data_set/flower_data/'
train_dir = image_path + 'train'
validation_dir = image_path + 'val'


if not os.path.exists('save_weights'):
    os.mkdirs('save_weights')

img_height, img_width = 224, 224
batch_size = 32
epochs = 10


train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           class_mode='categorical')
valid_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                batch_size=batch_size,
                                                                shuffle=False,
                                                                target_size=(img_height, img_width),
                                                                class_mode='categorical')
total_train, total_valid = train_data_gen.n, valid_data_gen.n


class_indices = train_data_gen.class_indices
inverse_dict = dict((v, k) for k, v in class_indices.items())
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)



model = AlexNet_v2(class_num=5)
model.build((batch_size, img_height, img_width, 3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learnning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=valid_data_gen,
                    validation_steps=total_valid // batch_size,
                    callbacks=callbacks)


# plot loss and accuracy image
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# figure 1
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# figure 2
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()