import ctypes
import os
from SSD.ssd_utils import *

from SSD.ssd_model import SSD
from SSD.ssd_utils import L2Norm

from CvT.models.cvt import *
from SSD.ssd_utils import PriorBox

USE224 = True


class CvT_SSD(SSD):
    """基于vgg-SSD进行特性功能重写"""

    def __init__(self, base_network, num_classes, mode='train'):
        """
            基于base_network的stages构建ssd
        :param base_network: feature-extractor模块列表
        :param mode: 模式
        :param num_classes: 这里的类别是真实类别+背景 ,共N+1
        """
        self.mode = mode
        loc_layers = []
        conf_layers = []
        base_net_out_channels = [viT.get_out_channels for viT in base_network]
        base_pBC = [4, 4, 6]  # priorBoxCountPerPixel for base-net
        assert len(base_pBC) == len(base_net_out_channels) == len(base_network), "base_network layer number not " \
                                                                                 "equal to base-pBC "
        l2Norms = [L2Norm(oc, 20) for oc in base_net_out_channels[:-1]]
        L2NormMaps = {idx: layer for idx, layer in enumerate(l2Norms)}
        for idx, oc in enumerate(base_net_out_channels):
            loc_layers.append(Conv2d(oc, base_pBC[idx] * 4, kernel_size=(3, 3), padding=(1, 1)))
            conf_layers.append(Conv2d(oc, base_pBC[idx] * num_classes, kernel_size=(3, 3), padding=(1, 1)))
        extra_pBC = [6, 6, 4, 4]  # priorBoxCountPerPixel for extra-net
        extra_layers = self.get_extra_layers(base_net_out_channels[-1])
        assert len(extra_pBC) * 2 == len(extra_layers), "extra_layers not be two ratio to extra-priorBoxCountPerPixel."
        for idx, layer in enumerate(extra_layers[1::2]):
            loc_layers.append(Conv2d(layer.out_channels, extra_pBC[idx] * 4, kernel_size=(3, 3), padding=(1, 1)))
            conf_layers.append(Conv2d(layer.out_channels, extra_pBC[idx] * num_classes, kernel_size=(3, 3), padding=(
                1, 1)))
        super(CvT_SSD, self).__init__(base_network=base_network, extra_layers=extra_layers, loc_layers=loc_layers,
                                      conf_layers=conf_layers, mode=self.mode, num_classes=num_classes,
                                      l2NormMaps=L2NormMaps)
        self.l2Norms = ModuleList(l2Norms)
        print('CvT-SSD模型初始化完毕.')

    def forward(self, x):
        """SSD计算预测特征图的物体类别和位置"""
        feature_maps = list()
        loc_pred = list()
        conf_pred = list()
        if self.l2NormMaps:  # 指定了归一化层和位置
            for idx, base_layer in enumerate(self.base_network):
                x, cls_token = base_layer(x)
                if idx in self.l2NormMaps:
                    feature_maps.append(self.l2NormMaps[idx](x))
                elif idx == len(self.base_network)-1:
                    feature_maps.append(x)
        else:  # 未指定归一化层则只使用最后一层
            for base_layer in self.base_network:
                x = base_layer(x)
            feature_maps.append(x)
        for idx, layer in enumerate(self.extra_layers):
            x = F.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                feature_maps.append(x)
        for (fm, ll, cl) in zip(feature_maps, self.loc_layers, self.conf_layers):
            loc_pred.append(ll(fm).permute(0, 2, 3, 1).contiguous())
            conf_pred.append(cl(fm).permute(0, 2, 3, 1).contiguous())
        loc_pred = torch.cat([pre_loc.view(pre_loc.shape[0], -1) for pre_loc in loc_pred], 1)  # 对每张图片拼接所有卷积层的位置预测
        conf_pred = torch.cat([pre_conf.view(pre_conf.shape[0], -1) for pre_conf in conf_pred], 1)  # 对每张图片拼接所有卷积层的类别预测

        if self.mode != "train":
            with torch.no_grad():
                return self.detector(loc_pred.view(loc_pred.shape[0], -1, 4),
                                     self.softmax(conf_pred.view(conf_pred.shape[0], -1, self.num_classes)),
                                     self.prior_boxes)
        else:
            return loc_pred.view(loc_pred.shape[0], -1, 4), conf_pred.view(conf_pred.shape[0], -1,
                                                                           self.num_classes), self.prior_boxes.unsqueeze(0)

    @staticmethod
    def get_extra_layers(in_channels):  # in_channels:前一层的输出通道
        """
            在骨干网络末端追加几层卷积用来获取预测值并拼接
            参考项目根目录下 introduce/SSD从1X1X1024后面的8次卷积.png.
        """
        if USE224:
            return [Conv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1)),  # Convs_224
                    Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 1
                    Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                    Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 2
                    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                    Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 3
                    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                    Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 1))  # 4
                    ]
        else:
          return [Conv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1)),  # Convs_384
                Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 1
                Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 2
                Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 3
                Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))  # 4
                ]

    def get_prior_boxes(self):
        """
            CvT-W24 的三个stage的特征图的大小是94->48->24 ,如果后接4个层,应该为 12, 6, 3, 1
            vgg-ssd: 特征图为 38->19->10->5->3->1
            即使CvT输入要求是384,224,但是我们VOC图片实际大小是300*300,这里的候选框也是按照300来
        """
        try:
            PRIOR_BOX_CONFIG_384 = {
                'MIN_DIM': 384,
                'ASPECT_RATIOS': [[2], [2], [2, 3], [2, 3], [2, 3], [2], [2]],
                'VARIANCE': [0.1, 0.2],
                'FEATURE_MAPS': [94, 48, 24, 12, 6, 3, 1],  # 384 前三位,224前三位: 56, 28, 14
                'MIN_SIZES': [21, 42, 63, 114, 163, 214, 265],
                'MAX_SIZES': [42, 63, 114, 163, 214, 265, 315],
                'STEPS': [3, 6, 13, 25, 50, 100, 300],
                'CLIP': True  # 一定要clip为 if=>真,不然越界了
            }
            PRIOR_BOX_CONFIG_224 = {
                'MIN_DIM': 224,
                'ASPECT_RATIOS': [[2], [2], [2, 3], [2, 3], [2, 3], [2], [2]],
                'VARIANCE': [0.1, 0.2],
                'FEATURE_MAPS': [56, 28, 14, 7, 4, 2, 1],  # 不知道其他CvT模型还是前三位不
                'MIN_SIZES': [21, 42, 63, 114, 163, 214, 265],
                'MAX_SIZES': [42, 63, 114, 163, 214, 265, 315],
                'STEPS': [6, 11, 22, 43, 75, 150, 300],
                'CLIP': True  # 一定要clip为 if=>真,不然越界了
            }
            with t.no_grad():
                prior_boxes = PriorBox(PRIOR_BOX_CONFIG_224 if USE224 else PRIOR_BOX_CONFIG_384)()
                if t.cuda.is_available():
                    prior_boxes = prior_boxes.cuda()
                return prior_boxes
        except Exception as e:
            logging.error(f'ERROR in build prior_boxes for cvt-ssd.' + str(e.args))
            exit(-1)


def weights_init(layer):
    if isinstance(layer, t.nn.Conv2d):
        t.nn.init.xavier_uniform_(layer.weight.data)  # xavier_uniform 废弃了
        layer.bias.data.zero_()


def build_ssd_from_cvt(cvt_configs, cvt_model_file_path=None, cvt_activate_method=QuickGELU, cvt_norm=LayerNorm_,
                       cvt_rgb_channels=3, ssd_mode='train', model_path=None, num_classes: int = 21):

    cvt_network = ConvolutionVisionTransformer(model_configs=cvt_configs, mode='object-detection', norm=cvt_norm,
                                               activate_method=cvt_activate_method, rgb_channels=cvt_rgb_channels)
    if cvt_model_file_path and not model_path:
        if os.path.exists(cvt_model_file_path):
            # 加载base-network网络层参数
            print(f'loading from {cvt_model_file_path} to CvT_ssd base_network...')
            cvt_network.load_weights(pretrained_model_file=cvt_model_file_path)
            print('finished.')
        else:
            logging.error('Error in load CvT weight, no such weight file!{}'.format(cvt_model_file_path))
            exit(-1)
    if t.cuda.is_available():
        cvt_network.to(t.device('cuda'))
    ssd_base_layers = cvt_network.get_ssd_base_layers()
    model = CvT_SSD(base_network=ssd_base_layers, num_classes=num_classes, mode=ssd_mode)
    if cvt_model_file_path and not model_path:
        # 初始化SSD网络层参数
        print('initial conf-extra-loc layers...')
        model.loc_layers.apply(weights_init)
        model.conf_layers.apply(weights_init)
        model.extra_layers.apply(weights_init)
        print('finished.')
    if model_path:
        if os.path.exists(model_path):
            # 加载base-network网络层参数
            print(f'loading from {model_path} to CvT_ssd network...')
            model.load_weights(model_file=model_path)
            print('finished.')
        else:
            logging.error('Error in load CvT_SSD weight, no such weight file!{}'.format(model_path))
            exit(-1)
    if t.cuda.is_available():
        model = model.cuda()
    return model
