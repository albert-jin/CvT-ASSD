import logging
import os
import torch.nn as nn
from .ssd_utils import *


class SSD(nn.Module):
    """SSD 端到端物体探测模型框架 默认feature-extract-base-network: VGG,可通过base_network指定其他骨干网络"""
    def __init__(self, base_network, extra_layers, loc_layers,conf_layers, num_classes =21, mode='train',size =300, l2Norm =None):
        super(SSD, self).__init__()
        self.base_network = nn.ModuleList(base_network)
        self.extra_layers = nn.ModuleList(extra_layers)
        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)
        self.num_classes = num_classes
        self.detector = Detector(self.num_classes, 0, 200, 0.01, 0.45)
        self.prior_boxes=self.get_prior_boxes()
        assert mode in ['train','test','val'], 'Error: only train,test,val can be set-mode!'
        self.mode = mode  # 运行模式,只能为训练/测试
        assert size == 300, f'ERROR: You specified size {self.mode} currently only SSD300 (size=300) is supported!'
        self.size = size
        self.l2Norm = l2Norm or L2Norm(512, 20)
        self.softmax = nn.Softmax(dim=-1)
    
    def get_prior_boxes(self):
        dataset_name = 'VOC' if self.num_classes == 21 else "COCO"
        try:
            PROIR_BOX_CONFIG = YAML_CONFIG['DATA'][dataset_name]['PRIOR_BOX']
            with torch.no_grad():
                return torch.autograd.Variable(PriorBox(PROIR_BOX_CONFIG)())
        except Exception as e:
            logging.error(f'ERROR in directory/global_configs.yaml DATA/{dataset_name}/PRIOR_BOX .\t{e.args}')
            exit(-1)

    def forward(self, x):
        """SSD计算预测特征图的物体类别和位置"""
        feature_maps = list()
        loc_pred = list()
        conf_pred = list()
        for idx in range(23):
            x = self.base_network[idx](x)
        feature_maps.append(self.l2Norm(x))
        for idx in range(23, len(self.base_network)):
            x = self.base_network[idx](x)
        feature_maps.append(x)
        for idx, layer in enumerate(self.extra_layers):
            x = F.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                feature_maps.append(x)
        for (fm, ll, cl) in zip(feature_maps, self.loc_layers, self.conf_layers):
            loc_pred.append(ll(x).permute(0, 2, 3, 1).contiguous())
            conf_pred.append(cl(x).permute(0, 2, 3, 1).contiguous())
        loc_pred = torch.cat([o.view(o.shape[0], -1) for o in loc_pred], 1)
        conf_pred = torch.cat([o.view(o.shape[0], -1) for o in conf_pred], 1)
        
        if self.mode != "train":
            with torch.no_grad():
                return self.detector(loc_pred.view(loc_pred.shape[0], -1, 4), self.softmax(conf_pred.view(conf_pred.shape[0], -1,self.num_classes)), self.priors.type(type(x.data)))
        else:
            return loc_pred.view(loc_pred.shape[0], -1, 4), conf_pred.view(conf_pred.shape[0], -1, self.num_classes), self.priors
        
    def load_weights(self, model_file:str):
        """从文件导入模型参数"""
        if model_file.endswith('.pkl') or model_file.endswith('.pth'):
            logging.info('Loading weights into ssd model...')
            self.load_state_dict(torch.load(model_file,
                                            map_location=lambda storage, loc: storage))
            logging.info('Finished loading ssd weights.')
        else:
            logging.error('Sorry only .pth and .pkl files supported!')
            exit(1)



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

    def vgg_ssd_multi_box(self):
        """output: 位置得分层和类别得分层"""
        priorBoxCountPerPixel = [4, 6, 6, 6, 4, 4]  # 每个点先验框的个数
        conv1_oc = self.vgg_base_layers[21].out_channels
        conv2_oc = self.vgg_base_layers[-2].out_channels
        loc_layers =[
            nn.Conv2d(conv1_oc,priorBoxCountPerPixel[0]*4,kernel_size=(3,3),padding=(1,1)),
            nn.Conv2d(conv2_oc, priorBoxCountPerPixel[1] * 4, kernel_size=(3, 3), padding=(1, 1)),
        ]
        conf_layers =[
            nn.Conv2d(conv1_oc, priorBoxCountPerPixel[0]*self.num_classes, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.Conv2d(conv2_oc, priorBoxCountPerPixel[1] * self.num_classes, kernel_size=(3, 3),
                      padding=(1, 1))
        ]
        for index,layer in enumerate(self.vgg_base_layers[1::2], 2):
            loc_layers.append(nn.Conv2d(layer.out_channels, priorBoxCountPerPixel[index] * 4, kernel_size=(3,3), padding=(1, 1)))
            conf_layers.append(nn.Conv2d(layer.out_channels,priorBoxCountPerPixel[index] *self.num_classes, kernel_size=(3,3), padding=(1, 1)))
        return loc_layers,conf_layers

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

    def get_vgg_extras_layers(self, in_channels = 1024):  # 最后一层的通道默认1024
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

def build_ssd_from_vgg():
