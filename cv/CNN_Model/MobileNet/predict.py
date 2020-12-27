from model import MobileNetV2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf


def run():
    img_height = 224
    img_width = 224

    # load image
    img = Image.open(img_path)
    # resize image to 224x224
    img = img.resize((img_width, img_height))
    plt.imshow(img)

    # scaling pixel value to (-1,1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    try:
        json_file = open(class_indices_path, 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    model = MobileNetV2(num_classes=5)
    # model.build((None, 224, 224, 3))  # when using subclass model
    model.load_weights(weights_path)
    result = np.squeeze(model.predict(img))
    prediction = tf.keras.layers.Softmax()(result).numpy()
    predict_class = np.argmax(result)
    print(class_indict[str(predict_class)], prediction[predict_class])
    plt.show()


if __name__ == '__main__':
    img_path = '../flower_photos/roses/12240303_80d87f77a3_n.jpg'
    weights_path = '../save_weights/MobileNetV2.ckpt'
    class_indices_path = '../class_indices.json'
    run()