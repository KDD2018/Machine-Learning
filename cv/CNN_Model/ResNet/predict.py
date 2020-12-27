from model import resnet50
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf


def predict(image_path, weights_path, class_indices_path):
    # load image
    img = Image.open(image_path)
    # resize image to 224x224
    img_height = img_width = 224
    img = img.resize((img_width, img_height))
    plt.imshow(img)
    # scaling pixel value to (0-1)
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    img = np.array(img).astype(np.float32)
    img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    try:
        json_file = open(class_indices_path, 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    feature = resnet50(img_height, img_width, num_class=5, include_top=False)
    # feature.trainable = False
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(5),
                                 tf.keras.layers.Softmax()])
    # model.build((None, 224, 224, 3))  # when using subclass model
    model.load_weights(weights_path)
    result = model.predict(img)
    prediction = np.squeeze(result)
    predict_class = np.argmax(result)
    print(class_indict[str(predict_class)], prediction[predict_class])
    plt.show()


if __name__ == '__main__':
    image_path = '../flower_photos/roses/12240303_80d87f77a3_n.jpg'
    weights_path = '../save_weights/resNet_50.ckpt'
    class_indices_path = '../class_indices.json'
    predict(image_path, weights_path, class_indices_path)