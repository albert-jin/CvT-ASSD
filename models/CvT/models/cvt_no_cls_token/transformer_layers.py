import logging
import torch as t
from torch.nn import Module, LayerNorm, GELU, Linear, Dropout, Sequential, BatchNorm2d, Conv2d, AvgPool2d, functional, \
    ModuleList, init, Identity
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
import numpy as np


class QuickGELU(GELU):
    def forward(self, inp_: t.Tensor) -> t.Tensor:
        return inp_ * t.sigmoid(1.702 * inp_)


class LayerNorm_(LayerNorm):
    def forward(self, inp_: t.Tensor) -> t.Tensor:
        inp_type = inp_.type()
        return (super().forward(inp_.type(t.FloatTensor))).type(inp_type)


class MLP(Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 activate_layer=GELU,
                 drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = activate_layer()
        out_features = out_features or in_features
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(Module):
    """注意力机制"""

    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., method='dw_bn',
                 kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.conv_proj_q = self._build_projection(dim_in, kernel_size, padding_q, stride_q,
                                                  'linear' if method == 'avg' else method)
        self.conv_proj_k = self._build_projection(dim_in, kernel_size, padding_kv, stride_kv,
                                                  method)
        self.conv_proj_v = self._build_projection(dim_in, kernel_size, padding_kv, stride_kv,
                                                  method)
        self.proj_q = Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = Linear(dim_in, dim_out, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim_out, dim_out)
        self.proj_drop = Dropout(proj_drop)

    # OK
    @staticmethod
    def _build_projection(dim_in, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            return Sequential(OrderedDict([
                ('conv', Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, bias=False,
                                groups=dim_in)),
                ('bn', BatchNorm2d(dim_in)),
                ('rearrange', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            return Sequential(OrderedDict([
                ('avg', AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True)),
                ('rearrange', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            return None
        else:
            raise ValueError('Unknown method ({})'.format(method))

    def forward(self, x, h, w):
        # x => shape: b (h w) c
        if self.conv_proj_q or self.conv_proj_k or self.conv_proj_v:
            q, k, v = self.conv_proj_q(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)) if self.conv_proj_q else x, \
                      self.conv_proj_k(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)) if self.conv_proj_k else x, \
                      self.conv_proj_v(rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)) if self.conv_proj_v else x
            # q,k,v => shape: b (h w) c
            q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
            k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
            v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
            attn_value = t.einsum('bhlk, bhtk -> bhlt', q, k) * self.scale
            attn_score = functional.softmax(attn_value, dim=-1)
            attn_score = self.attn_drop(attn_score)
            x = t.einsum('bhlt, bhtv -> bhlv', attn_score, v)
            x = rearrange(x, 'b h t d -> b t (h d)')
        return self.proj_drop(self.proj(x))


# OK
class ConvEmbed(Module):
    """图片卷积embedding"""

    def __init__(self, patch_size: tuple = (7, 7), in_channels=3, embedding_dim=64, stride: tuple = (4, 4),
                 padding=(2, 2),
                 norm=None):
        super(ConvEmbed, self).__init__()
        self.norm = norm
        #  Conv2d  (input_channel,output_channel,kernel_size(卷积核长宽),步长,padding,
        #  dilation(空洞卷积,扩张操作：控制kernel里面点（卷积核）的间距,默认1),groups(分组卷积,默认一组),
        #  bias(卷积的还可在输出加个可学习的方差,默认True))
        # input_:(N,C,H,W) output_:(N,C,H out,W out)
        self.conv2d_ = Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=stride, padding=padding)
        # 对 输入在最后一维(维度大小embedding_dim) 进行归一化
        # 使得最后一维(channel)所有数据满足均值0，方差1
        # input_:B, H, W, C => B, H, W, C
        if self.norm:
            self.norm_layer = self.norm(embedding_dim)

    def forward(self, x):
        # n c h w => n c h (out) w (out)
        x = self.conv2d_(x)
        if self.norm:
            n, c, h, w = x.shape
            x = self.norm_layer(rearrange(x, 'n c h w -> n (h w) c'))
            x = rearrange(x, 'n (h w) c -> n c h w', h=h, w=w)
        return x


class VisionTransformer(Module):
    def __init__(self, patch_size=(16, 16), patch_stride=(16, 16), patch_padding=0, in_channels=3, embedding_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.
                 , act_layer=GELU, norm=LayerNorm, **kwargs):
        super(VisionTransformer, self).__init__()
        self.patch_embed = ConvEmbed(patch_size=patch_size, in_channels=in_channels, stride=patch_stride,
                                     padding=patch_padding, embedding_dim=embedding_dim, norm=norm)
        self.pos_drop = Dropout(p=drop_rate)
        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = ModuleList([Block(dim_in=embedding_dim, dim_out=embedding_dim, num_heads=num_heads,
                                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                                        attn_drop=attn_drop_rate, drop_path=drop_path, act_layer=act_layer,
                                        norm=norm, **kwargs) for drop_path in dpr])
        self.apply(self.init_weights_trunc_normal)

    @staticmethod
    def init_weights_trunc_normal(layer):
        if isinstance(layer, Linear):
            trunc_normal_(layer.weight, std=0.02)
            if layer.bias:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, (LayerNorm, BatchNorm2d)):
            init.constant_(layer.bias, 0)
            init.constant_(layer.weight, 1.0)

    def forward(self, x):
        """input: n c h w => output: n c h (out) w (out)"""
        x = self.patch_embed(x)  # n c h w => n c h (out) w (out)
        n, c, h, w = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, h, w)
        return rearrange(x, 'n (h w) c -> n c h w', h=h, w=w)


class Block(Module):
    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0.0, drop=0.0, drop_path=0.0,
                 mlp_ratio=4, activate_layer=GELU, norm=LayerNorm, **kwargs):
        super().__init__()
        self.norm_layer1 = norm(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm_layer2 = norm(dim_out)
        self.mlp = MLP(in_features=dim_out, hidden_features=int(dim_out * mlp_ratio), activate_layer=activate_layer,
                       drop=drop)

    def forward(self, x, h, w):
        x_ = self.norm_layer1(x)
        attention_ = self.attn(x_, h, w)
        x_ = x + self.drop_path(attention_)
        return x_ + self.drop_path(self.mlp(self.norm_layer2(x_)))
