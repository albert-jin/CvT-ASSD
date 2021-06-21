from torch import nn
class vgg_layers(object):  # model:vgg_d
    """
        vgg300 VGG16的D模型
        项目根目录下introduce/VGG16的D模型.png 了解更多.
    """

    def __init__(self, num_classes):
        super(vgg_layers, self).__init__()
        self.num_classes = num_classes
        self.vgg_base_layers = self.get_vgg_base_layers()
        self.vgg_extras_layers = self.get_vgg_extras_layers()
        self.loc_layers, self.conf_layers =self.vgg_ssd_multi_box()

    def __call__(self):
        """get four layers:base_layers、extras_layers、loc_layers、conf_layers"""
        return self.vgg_base_layers, self.vgg_extras_layers, self.loc_layers, self.conf_layers

    def vgg_ssd_multi_box(self):
        """output: 位置得分层和类别得分层"""
        priorBoxCountPerPixel = [4, 6, 6, 6, 4, 4]  # 每个点先验框的个数
        conv1_oc = self.vgg_base_layers[21].out_channels  # 512
        conv2_oc = self.vgg_base_layers[-2].out_channels  # 1024
        loc_layers = [
            nn.Conv2d(conv1_oc, priorBoxCountPerPixel[0] * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(conv2_oc, priorBoxCountPerPixel[1] * 4, kernel_size=(3, 3), padding=(1, 1)),
        ]
        conf_layers = [
            nn.Conv2d(conv1_oc, priorBoxCountPerPixel[0] * self.num_classes, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.Conv2d(conv2_oc, priorBoxCountPerPixel[1] * self.num_classes, kernel_size=(3, 3),
                      padding=(1, 1))
        ]
        for index, layer in enumerate(self.vgg_extras_layers[1::2], 2):
            loc_layers.append(
                nn.Conv2d(layer.out_channels, priorBoxCountPerPixel[index] * 4, kernel_size=(3, 3), padding=(1, 1)))
            conf_layers.append(
                nn.Conv2d(layer.out_channels, priorBoxCountPerPixel[index] * self.num_classes, kernel_size=(3, 3),
                          padding=(1, 1)))
        return loc_layers, conf_layers

    def get_vgg_base_layers(self, in_channels=3, use_batch_norm=False):  # 默认vgg 输入channels 是RGB 3通道
        """
            列表顺序存储的VGG网络的网络层
        """
        layers = [nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
                  nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.ReLU(inplace=True)]
        if use_batch_norm:
            layers_with_norm = []
            for layer in layers:
                layers_with_norm.append(layer)
                if isinstance(layer, nn.Conv2d):
                    layers_with_norm.append(nn.BatchNorm2d(layer.out_channels))
            layers = layers_with_norm
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
        conv7 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])
        return layers

    def get_vgg_extras_layers(self, in_channels=1024):  # 最后一层的通道默认1024
        """
            在骨干网络末端追加几层卷积用来获取预测值并拼接
            项目根目录下 introduce/SSD从1X1X1024后面的8次卷积.png 了解更多.
        """
        return [nn.Conv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))]
