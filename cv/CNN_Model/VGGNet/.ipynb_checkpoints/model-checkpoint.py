from tensorflow.keras import layers, models, Model, Sequential


class VGGNet(object):
    def __init__(self, img_height, img_width, class_num, name):
        self.img_height = img_height
        self.img_width = img_width
        self.class_num = class_num
        self.name = name

    def vgg_arch(self, feature):
        """
        VGG Model核心模块
        :param feature: 特征层
        :return: Model
        """
        input_image = layers.Input(shape=(self.img_height, self.img_width, 3), dtype='float32')
        x = feature(input_image)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dense(self.class_num)(x)
        output = layers.Softmax()(x)
        model = models.Model(inputs=input_image, outputs=output)
        return model

    def features(self, cfg):
        """
        VGG Modle特征层
        :param cfg: 特征层参数配置
        :return: 特征层
        """
        feature_layers = []
        for v in cfg:
            if v == "M":
                feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
            else:
                conv2d = layers.Conv2D(v, kernel_size=3, padding='same', activation='relu')
                feature_layers.append(conv2d)
        return Sequential(feature_layers, name='feature')

    def vgg(self):
        """
        VGG Model
        :param model_name:VGG系列模型的名称
        :return: model
        """
        try:
            cfg = cfgs[self.name]
        except:
            print(f'Warning: model number {self.name} not supported.')
            exit(-1)
        model = self.vgg_arch(self.features(cfg))

        return model


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}