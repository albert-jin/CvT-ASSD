import torch
from torch.nn import Module, LayerNorm, GELU

class ConvolutionVisionTransformer(Module):
    def __init__(self, rgb_channels =3, activate_layer =GELU):
        super(ConvolutionVisionTransformer, self).__init__()