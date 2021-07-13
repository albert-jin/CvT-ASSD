import collections

import torch.nn as nn

from SSD.ssd_utils import *


class SSD(nn.Module):
    """SSD 端到端物体探测模型框架 默认feature-extract-base-network: VGG,可通过base_network指定其他骨干网络"""

    def __init__(self, base_network, extra_layers, loc_layers, conf_layers, num_classes=21, mode='train', size=300,
                 l2NormMaps: dict = None):
        super(SSD, self).__init__()
        self.base_network = nn.ModuleList(base_network)
        self.extra_layers = nn.ModuleList(extra_layers)
        assert len(loc_layers) == len(conf_layers), "位置层应和类别层层数相同!"
        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)
        self.num_classes = num_classes
        self.prior_boxes = self.get_prior_boxes()
        assert mode in ['train', 'test', 'val'], 'Error: only train,test,val can be set-mode!'
        self.mode = mode  # 运行模式,只能为训练/测试
        '''size 为模型的输入长宽,只有在vgg_ssd里面需强制为300'''
        # assert size == 300, f'ERROR: You specified size {self.mode} currently only SSD300 (size=300) is supported!'
        # self.size = size
        self.l2NormMaps = l2NormMaps
        if 22 in self.l2NormMaps:
            self.l2Norm = self.l2NormMaps[22]
        # only use softmax &detector when in test mode.
        self.softmax = nn.Softmax(dim=-1)
        self.detector = Detector(self.num_classes, 200, 0.01, 0.45)

    def get_prior_boxes(self):
        dataset_name = 'VOC' if self.num_classes == 21 else "COCO"
        try:
            PRIOR_BOX_CONFIG = YAML_CONFIG['DATA'][dataset_name]['PRIOR_BOX']
            with torch.no_grad():
                prior_boxes = PriorBox(PRIOR_BOX_CONFIG)()
                if torch.cuda.is_available():
                    prior_boxes = prior_boxes.cuda()
                return prior_boxes
        except Exception as e:
            logging.error(f'ERROR in directory/data_configs.yaml DATA/{dataset_name}/PRIOR_BOX .\t{e.args}')
            exit(-1)

    def forward(self, x):
        """SSD计算预测特征图的物体类别和位置"""
        feature_maps = list()
        loc_pred = list()
        conf_pred = list()
        if self.l2NormMaps:  # 指定了归一化层和位置
            for idx, base_layer in enumerate(self.base_network):
                x = base_layer(x)
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
                                                                           self.num_classes), self.prior_boxes

    def load_weights(self, model_file: str):
        """从文件导入模型参数"""
        if model_file.endswith('.pkl') or model_file.endswith('.pth'):
            logging.info('Loading weights into ssd model...')
            params = torch.load(model_file, map_location=lambda storage, loc: storage)
            self.load_state_dict(params)
            logging.info('Finished loading ssd weights.')
        else:
            logging.error('Sorry only .pth and .pkl files supported!')
            exit(1)

    def load_weights_from_old(self, model_file: str):
        """从文件导入模型参数"""
        if model_file.endswith('.pkl') or model_file.endswith('.pth'):
            logging.info('Loading weights into ssd model...')
            state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            new_state_dict = collections.OrderedDict()
            key_map = {'vgg': 'base_network', 'extras': 'extra_layers', 'loc': 'loc_layers', 'conf': 'conf_layers',
                       'L2Norm': 'l2Norm'}
            for layer_name, params in state_dict.items():
                change_flag = False
                for key in key_map:
                    if key in layer_name:
                        new_state_dict[layer_name.replace(key, key_map[key])] = params
                        change_flag = True
                        break
                if not change_flag:
                    new_state_dict[layer_name] = params
            assert len(new_state_dict) == len(state_dict)  # 保证转换后参数大小一致
            self.load_state_dict(new_state_dict)
            logging.info('Finished loading ssd weights.')
        else:
            logging.error('Sorry only .pth and .pkl files supported!')
            exit(1)
