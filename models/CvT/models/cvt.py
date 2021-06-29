from .transformer_layers import *


class ConvolutionVisionTransformer(Module):
    """图片分类以及物体检测base-network"""
    def __init__(self, model_configs, mode: str = 'classifier', rgb_channels=3, activate_method=GELU, norm=LayerNorm):
        super(ConvolutionVisionTransformer, self).__init__()
        if mode not in ['classifier', 'object-detection']:
            logging.error('ConvolutionVisionTransformer only supports classifier,object-detection now!')
            exit(-1)
        self.mode = mode
        configs = model_configs['SPEC']
        self.num_stages = configs['NUM_STAGES']
        in_channels = rgb_channels
        for idx in range(self.num_stages):
            vit_configs = {'patch_size': (configs['PATCH_SIZE'][idx],configs['PATCH_SIZE'][idx]),
                           'patch_padding': configs['PATCH_PADDING'][idx], 'embedding_dim': configs['DIM_EMBED'][idx],
                           'depth': configs['DEPTH'][idx], 'num_heads': configs['NUM_HEADS'][idx],
                           'mlp_ratio': configs['MLP_RATIO'][idx], 'qkv_bias': configs['QKV_BIAS'][idx],
                           'drop_rate': configs['DROP_RATE'][idx], 'attn_drop_rate': configs['ATTN_DROP_RATE'][idx],
                           'drop_path_rate': configs['DROP_PATH_RATE'][idx], 'stride_q': configs['STRIDE_Q'][idx],
                           'with_cls_token': configs['CLS_TOKEN'][idx], 'patch_stride': configs['PATCH_STRIDE'][idx],
                           'method': configs['QKV_PROJ_METHOD'][idx], 'kernel_size': configs['KERNEL_QKV'][idx],
                           'padding_q': configs['PADDING_Q'][idx], 'padding_kv': configs['PADDING_KV'][idx],
                           'stride_kv': configs['STRIDE_KV'][idx]
                           }
            setattr(self, f'stage{idx}', VisionTransformer(in_channels=in_channels, activate_method=activate_method,
                                                           norm=norm, **vit_configs))
            in_channels = configs['DIM_EMBED'][idx]
        self.norm = norm(in_channels)
        num_classes = configs['NUM_CLASSES'] if 'NUM_CLASSES' in configs else 0
        if self.mode == 'classifier':
            self.head = Linear(in_channels, num_classes) if num_classes > 0 else Identity
            trunc_normal_(self.head.weight, std=0.02)

    def load_weights(self, pretrained_model_file):
        """从文件导入模型参数"""
        if pretrained_model_file.endswith('.pkl') or pretrained_model_file.endswith('.pth'):
            logging.info('Loading weights into CvT...')
            state_dict = t.load(pretrained_model_file, map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            key_map = {}
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
            self.load_state_dict(new_state_dict, strict=False)
            logging.info('Finished loading CvT weights.')
        else:
            logging.error('Sorry only .pth and .pkl files supported!')
            exit(1)

    def forward(self, x):
        for idx in range(self.num_stages):
            x, cls_token = getattr(self, f'stage{idx}')(x)
        x = t.squeeze(self.norm(cls_token))
        if self.mode == 'classifier':
            return self.head(x)
        return x

    def get_ssd_base_layers(self):
        cvt_base_layers = [getattr(self, f'stage{idx}') for idx in range(self.num_stages)]
        return cvt_base_layers
