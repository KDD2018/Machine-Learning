from tensorflow.keras import layers, models, Model, Sequential


class Inception(layers.Layer): # 128, 128, 192, 32, 96, 64
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs) # 28 x 28 x 256
        self.branch1 = Sequential([
            layers.Conv2D(ch1x1, kernel_size=1, use_bias=False, name='conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn'),
            layers.ReLU()
        ], name='branch1') # 28 x 28 x 128

        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, use_bias=False, name='0/conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='0/bn'),
            layers.ReLU(), # 28 x 28 x 128
            layers.Conv2D(ch3x3, kernel_size=3, padding='same', use_bias=False, name='1/conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='1/bn'),
            layers.ReLU() # 28 x 28 x 192
        ], name='branch2')

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, use_bias=False, name='0/conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='0/bn'),
            layers.ReLU(), # 28 x 28 x 32
            layers.Conv2D(ch5x5, kernel_size=3, padding='same', use_bias=False, name='1/conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='1/bn'),
            layers.ReLU() # 28 x 28 x 96
        ], name='branch3')

        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),  # default strides=pool_size  28 x 28 x 256
            layers.Conv2D(pool_proj, kernel_size=1, use_bias=False, name='1/conv'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='1/bn'),
            layers.ReLU() # 28 x 28 x 64
        ], name='branch4')

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(layers.Layer):
    def __init__(self, num_class, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, use_bias=False, name='conv/conv')
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv/bn')
        self.relu1 = layers.ReLU()

        self.fc1 = layers.Dense(1024, activation='relu', name='fc1')
        self.fc2 = layers.Dense(num_class, name='fc2')
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 X 14 x 14
        x = self.averagePool(inputs)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 X 4 x 4
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_class
        x = self.softmax(x)
        return x


def GoogLeNet(img_height=224, img_width=224, class_num=1000, aux_logits=False):
    """
    GoogLeNet网络
    :param img_height:
    :param img_width:
    :param class_num:
    :param aux_logits:
    :return:
    """
    input_image = layers.Input(shape=(img_height, img_width, 3), dtype='float32')
    # (None, 224, 224, 3)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv1/conv')(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1/bn')(x)
    x = layers.ReLU()(x)
    # (None, 112, 112, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_1')(x)
    # (None, 56, 56, 64)

    x = layers.Conv2D(64, kernel_size=1, use_bias=False, name='conv2/conv')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv2/bn')(x)
    x = layers.ReLU()(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(192, kernel_size=3, padding='same', use_bias=False, name='conv3/conv')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv3/bn')(x)
    x = layers.ReLU()(x)
    # (None, 56, 56, 192)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_2')(x)

    # (None, 28, 28, 192)
    x = Inception(64, 96, 128, 16, 32, 32, name='inception3a')(x)
    # (None, 28, 28, 256)
    x = Inception(128, 128, 192, 32, 96, 64, name='inception3b')(x)

    # (None, 28, 28, 480)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_3')(x)
    # (None, 14, 14, 480)
    x = Inception(192, 96, 208, 16, 48, 64, name='inception4a')(x)
    if aux_logits:
        aux1 = InceptionAux(class_num, name='aux1')(x)

    # (None, 14, 14, 512)
    x = Inception(160, 112, 224, 24, 64, 64, name='inception4b')(x)
    # (None, 14, 14, 512)
    x = Inception(128, 128, 256, 24, 64, 64, name='inception4c')(x)
    # (None, 14, 14, 512)
    x = Inception(112, 144, 288, 32, 64, 64, name='inception4d')(x)
    if aux_logits:
        aux2 = InceptionAux(class_num, name='aux2')(x)

    # (None, 14, 14, 528)
    x = Inception(256, 160, 320, 32, 128, 128, name='inception4e')(x)
    # (None, 14, 14, 832)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='maxpool_4')(x)

    # (None, 7, 7, 832)
    x = Inception(256, 160, 320, 32, 128, 128, name='inception5a')(x)
    # (None, 7, 7, 832)
    x = Inception(384, 192, 384, 48, 128, 128, name='inception5b')(x)
    # (None, 7, 7, 1024)
    x = layers.AvgPool2D(pool_size=7, strides=1, name='avgpool_1')(x)

    # (None, 1, 1, 1024)
    x = layers.Flatten(name='output_flatten')(x)
    # (None, 1024)
    x = layers.Dropout(rate=0.4, name='output_dropout')(x)
    x = layers.Dense(class_num, name='fc')(x)
    # (None, num_class)
    aux3 = layers.Softmax()(x)

    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model

