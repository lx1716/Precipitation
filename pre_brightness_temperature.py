import os
import re
import glob
import copy
import torch
import math

from torch import nn
import numpy as np
import pandas as pd
import collections.abc
import torch.nn.functional as F

from datetime import datetime, timedelta
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from einops import repeat, rearrange


def exists(val):
    return val is not None


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# time encoding
def time_encoding(init_time, num_steps, freq=6):
    '''
    This function generates time encodings for the given initial time,
    number of steps, and frequency of time intervals. The approach is
    based on the description provided in the paper
    "GraphCast: Learning skillful medium-range global weather forecasting"
    by DeepMind (https://arxiv.org/abs/2212.12794).

    init_time: e.g. pd.Timestamp('2019-01-01 00:00:00')
    num_steps: e.g. 60 (00ofD1,06ofD1,12ofD1,18ofD1,00ofD2,06ofD2,12ofD2,18ofD2,……,00ofD15,06ofD15,12ofD15,18ofD15)
    freq: e.g. 6 (hours)
    '''
    init_time = np.array([init_time])
    # init_time: e.g. array([Timestamp('2023-06-01 00:00:00')], dtype=object)
    tembs = []
    for i in range(num_steps):
        # use 3 times centered at the current time to encode the time
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        # hours: e.g. array([Timedelta('-1 days +18:00:00'), Timedelta('0 days 00:00:00'), Timedelta('0 days 06:00:00')], dtype=object)
        times = init_time[:, None] + hours[None]
        # times: e.g. array([[Timestamp('2023-05-31 18:00:00'), Timestamp('2023-06-01 00:00:00'),Timestamp('2023-06-01 06:00:00')]], dtype=object)
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        # times:e.g. Period('2023-05-31 18:00', 'H'), Period('2023-06-01 00:00', 'H'), Period('2023-06-01 06:00', 'H')
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        # times: e.g. [(0.4166666666666667, 0.0), (0.4193989071038251, 0.0), (0.4221311475409836, 0.0)]
        temb = torch.from_numpy(np.array(times, dtype=np.float32))
        # transform [(0.416, 0.0), (0.419, 0.0), (0.422, 0.0)]
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        # transform the float time value to sin and cos values
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    # tembs.shape: e.g. (60, 1, 12)
    # 60 represent the number of times(00:00:00, 06:00:00, ……),
    # 12 represent 3*4 features for each time(3 times centered at 00:00:00(or 06:00:00), 4 features for each time)
    return torch.stack(tembs)


class CubeEmbed(nn.Module):
    def __init__(
            self,
            in_chans=3,  # e.g. 75
            in_frames=1,  # e.g. 2
            patch_size=4,  # e.g. 4
            embed_dim=96,  # e.g. 1536
            norm_layer=None,
            flatten=False,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans * in_frames, embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=0
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x.shape=, e.g. (n, 2*70, 721, 1440)
        x = x.view(-1, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        if x.ndim == 5:
            x = rearrange(x, 'n t c h w -> n (t c) h w')
        # use a conv layer to surpress the input, e.g. (n, c=150, 721, 1440) -> (n, c=1536, 180, 360)
        # print(x.size())
        # x = x.view(x.shape[0], -1, x.shape[2], x.shape[3])

        x = self.proj(x)
        # flatten is true, e.g. (n, c=1536, 180, 360) -> (n, 180*360=64800, c=1536)
        if self.flatten:
            x = rearrange(x, 'n c h w -> n (h w) c')
            x = self.norm(x)
        return x


class PeriodicConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert max(self.padding) > 0

    def forward(self, x):
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(
            x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
        )
        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            temb_dim=0,
            dropout=0,
            scale_shift=False,
            conv_class=nn.Conv2d,
    ):
        super().__init__()

        self.scale_shift = scale_shift

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_dim,
                         eps=1e-6, affine=True),
            nn.SiLU(),
            conv_class(in_dim, out_dim, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, 2 * out_dim if scale_shift else out_dim)
        ) if temb_dim > 0 else nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_dim,
                         eps=1e-6, affine=True),
            nn.SiLU() if scale_shift else nn.Identity(),
            nn.Dropout(p=dropout),
            conv_class(out_dim, out_dim, 3, padding=1)
        )

        if in_dim == out_dim:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x, temb=None):
        h = self.in_layers(x)
        if temb is None:
            h = self.out_layers(h)
        else:
            temb = self.emb_layers(temb)
            temb = rearrange(temb, 'n c -> n c 1 1')
            if self.scale_shift:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = temb.chunk(2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + temb
                h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, use_conv=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2

        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        # x.shape:[1, c=1536, 180, 360]
        # use a conv layer to deduce h and w to half, e.g. (n, c=1536, 180, 360) -> (n, c=1536, 90, 180)
        return self.op(x)


class Upsample(nn.Module):
    def __init__(
            self,
            channels,
            out_channels=None,
            use_conv=False,
            use_deconv=False,
            conv_class=nn.Conv2d
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_deconv = use_deconv

        if use_deconv:
            self.op = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.op = conv_class(
                self.channels, self.out_channels, 3, padding=1)

    def forward(self, x, output_size=None):
        if self.use_deconv:
            return self.op(x)

        if output_size is None:
            y = F.interpolate(x.float(), scale_factor=2.0,
                              mode="nearest").to(x)
        else:
            y = F.interpolate(x.float(), size=tuple(
                output_size), mode="nearest").to(x)

        if self.use_conv:
            y = self.op(y)

        return y


class DownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_dim: int = 0,
            num_layers: int = 1,
            down=True,
            scale_shift=False,
            conv_class=nn.Conv2d,
    ):
        super().__init__()
        self.down = down

        if down:
            self.downsample = Downsample(
                in_channels, out_channels, use_conv=True)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                ResBlock(
                    out_channels if down else in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, temb=None):
        # x.shape:[1, c=1536, 180, 360]
        if self.down:
            # use a conv layer to deduce h and w to half, e.g. (n, c=1536, 180, 360) -> (n, c=1536, 90, 180)
            x = self.downsample(x)
        # use a res block to supress x and time embedding into together
        for blk in self.resnets:
            x = blk(x, temb)
        # x.shape:[1, c=1536, 90, 180]
        return x


class UpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            skip_channels: int,
            temb_dim: int = 0,
            num_layers: int = 1,
            up=True,
            scale_shift=False,
            conv_class=nn.Conv2d,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels + skip_channels if i == 0 else out_channels
            resnets.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.up = up
        if up:
            self.upsample = Upsample(
                out_channels, out_channels, use_deconv=True)

    def forward(self, x, temb=None, output_size=None):
        for blk in self.resnets:
            x = blk(x, temb)

        if self.up:
            x = self.upsample(x, output_size)
        return x


conv_class = PeriodicConv2d
norm_layer = partial(nn.LayerNorm, eps=1e-6)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformerV2(nn.Module):  # (nn.Module):
    def __init__(self, img_size=150, patch_size=16, in_chans=768, num_classes=1000, embed_dims=[768, 768, 768, 768],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[6, 6, 6, 6], num_stages=4, linear=False):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            # setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"norm{i + 1}", norm)
            # setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            # trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, H, W):
        # B = x.shape[0]
        # B, L, C = x.shape
        outs = []

        for i in range(self.num_stages):
            # patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            # x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x, H, W):
        x = self.forward_features(x, H, W)
        # x = self.head(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class H8Transformer(nn.Module):
    def __init__(
            self,
            in_chans,
            out_chans,
            in_frames,
            image_size,
            # window_size=8,
            patch_size=4,
            down_times=0,
            embed_dim=768,
            num_heads=[8, 8, 8, 8],
            depths=[6, 6, 6, 6],
            mlp_ratio=4,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times

        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=True,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(54, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if self.down_times > 0:
            down_blocks = []
            up_blocks = []
            for i in range(self.down_times):
                down_blocks.append(
                    DownBlock2D(
                        embed_dim,
                        embed_dim,
                        temb_dim=embed_dim,
                        num_layers=min(4, (i + 1) ** 2),
                        scale_shift=True,
                        conv_class=conv_class
                    )
                )
                up_blocks.append(
                    UpBlock2D(
                        embed_dim,
                        embed_dim,
                        embed_dim,
                        temb_dim=embed_dim,
                        num_layers=min(4, (self.down_times - i) ** 2),
                        scale_shift=True,
                        conv_class=conv_class
                    ),
                )

            self.down_blocks = nn.ModuleList(down_blocks)
            self.up_blocks = nn.ModuleList(up_blocks)

        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_layers, embed_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(embed_dim, out_chans * patch_size * patch_size)

        self.pvt = PyramidVisionTransformerV2(embed_dims=[embed_dim] * self.num_layers, num_heads=num_heads, depths=depths)

    def forward(self, input, temb=None, const=None, noise=None):
        # input.shape: (n, t=2, c=70, h, w)
        if exists(const):
            x = torch.cat([input, const], dim=2)
        else:
            x = input

        h = self.patch_embed(x)

        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)

        if exists(temb):
            # temb.shape: e.g. (i in range(60), 12), transform the shape from (1,60) to (1,c=1536)
            temb = self.time_embed(temb.to(h))
            temb = temb.view(-1, temb.shape[0])

        # unet down, this part is optinal, but is 1 in actual
        if self.down_times > 0:
            # to feed h into the DownBlock(CNN layer), we need to reshape h from (n, 64800, c=1536) back to (n, c=1536, 180, 360)
            # beacuase (n, 64800, c=1536) is suitable for SwinLayer, but (n, c=1536, 180, 360) is suitable for CNN layer
            h = rearrange(h, 'n (h w) c -> n c h w', h=self.feat_size[0])
            hs = []
            for blk in self.down_blocks:
                h = blk(h, temb)
                hs.append(h)
            feat_size = h.shape[-2:]
            h_size, w_size = h.shape[-2:]
            h = rearrange(h, 'n c h w -> n (h w) c')

        # attention (swin or vit)
        # outs = []
        # for i, blk in enumerate(self.layers):
        #     h = blk(h)
        #     out = getattr(self, f"norm{i}")(h)
        #     outs.append(out)
        outs = self.pvt(h, h_size, w_size)
        h = self.fpn(torch.cat(outs, dim=-1))

        # unet up
        if self.down_times > 0:
            h = rearrange(h, 'n (h w) c -> n c h w', h=feat_size[0])
            for blk in self.up_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = blk(h, temb)
            h = rearrange(h, 'n c h w -> n (h w) c')

        out = self.head(h)
        out = rearrange(out,
                        'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
                        h=self.feat_size[0],
                        p1=self.patch_size,
                        p2=self.patch_size,
                        )

        if out.shape[-2:] != input.shape[-2:]:
            out = F.interpolate(
                out.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)

        return out


class Predict_Net(nn.Module):

    def __init__(
            self,
            in_frames,
            out_frames,
            decoder,
            dtype=torch.float32,
    ):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        # self.step_range = step_range
        self.decoder = nn.ModuleList(decoder)
        # self.device = device
        self.dtype = dtype
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(1.0)

    # load the parameters into each decoder(Utransformer)
    def load(self, model_dir, fmt='pth'):
        import os
        from safetensors import safe_open
        from safetensors.torch import save_file
        # loop each decoder(short0-5, medium5-10, long10-15)
        for i, name in enumerate(['fuxi_short']):  # , 'fuxi_short', 'fuxi_short']):
            checkpoint = os.path.join(model_dir, f'{name}.{fmt}')
            print(f'load from {checkpoint} ...')
            # the checkpoint file may be saved in different format
            model_state = {}
            if fmt == "pth":
                chkpt = torch.load(checkpoint, map_location=torch.device("cpu"))
                for k, v in chkpt["model"].items():
                    k = k.replace('decoder.', '')
                    model_state[k] = v

                save_name = os.path.join(model_dir, f'{name}.st')
                print(f'save to {save_name} ...')
                save_file(model_state, save_name)


            elif fmt == "st":
                with safe_open(checkpoint, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        model_state[k] = f.get_tensor(k)
            # load the parameters into each decoder
            self.decoder[i].load_state_dict(model_state, strict=False)

        # mean, std and other 2 things are saved in buffer.ts
        buffer = os.path.join(model_dir, f'buffer.{fmt}')
        if os.path.exists(buffer):
            print(f'load from {buffer} ...')
            if fmt == 'pth':
                buffer = torch.load(buffer)
                for k, v in buffer.items():
                    self.register_buffer(k, v.to(self.device))
            elif fmt == 'st':
                with safe_open(buffer, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        v = f.get_tensor(k)
                        self.register_buffer(k, v.to(self.device))

    # normalize the input using Z-score method, inv is used to inverse the normalization
    def normalize(self, x, inv=False):
        # mean and std are from the buffer.ts
        mean = self.mean.to(self.dtype)
        std = self.std.to(self.dtype)
        # from normalized value back to original value
        if inv:
            x = x * std + mean
            tp = x[:, -1].exp() - 1
            x[:, -1] = tp.clamp(min=0)
        # from original value to normalized value
        else:
            tp = x[:, -1]
            tp = torch.log(1 + tp.clamp(min=0))
            x[:, -1] = tp
            x = (x - mean) / std
        return x

    # process the input
    def process_input(self, input, hw=(2400, 2400)):
        # push the input.shape to be (,,720, 1440), ???why not (,,721, 1440)???
        if input.shape[-2:] != hw:
            input = F.interpolate(
                input,
                size=hw,
                mode="bilinear",
                align_corners=False
            )

        return input

    # process the output
    def process_output(self, output, hw=(2401, 2401)):
        # push the output.shape to be (,,721, 1440)
        output = F.interpolate(
            output,
            size=hw,
            mode="bilinear",
            align_corners=False
        )
        # from normalized value back to original value
        # output = self.normalize(output, inv=True)
        return output

    # @torch.no_grad()
    def forward(self, input, tembs, outputs_template=None):
        # num_steps = sum(self.step_range)

        input = self.process_input(input, hw=(2400, 2400))
        # print(input.size())
        # input = F.interpolate(input, size=(256, 256), mode='bilinear', align_corners=False)
        decoder = self.decoder[0]
        const = None

        output = self.process_output(decoder(input, temb=tembs[0], const=const))
        return output

def build():

    # device = torch.device(args.device)

    # hyper parameters
    in_chans = 7
    in_frames = 2
    out_frames = 1
    out_chans = 7
    embed_dim = 768  # 1536
    num_heads = [8, 8, 8, 8]  # 24
    depths = [2, 2, 4, 2]  # [12, 12, 12, 12]
    image_size = (2400, 2400)  # (720, 1440)
    # image_size = (256, 256)
    patch_size = 4
    down_times = 2
    # window_size = 6  # 9

    decoder = H8Transformer(
        in_chans=in_chans,
        out_chans=out_chans,
        in_frames=in_frames,
        image_size=image_size,
        # window_size=window_size,
        patch_size=patch_size,
        down_times=down_times,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
    )

    model = Predict_Net(
        in_frames=in_frames,
        out_frames=out_frames,
        decoder=[decoder],  # decoder, decoder],
    )
    # step = 4099,
    return model

def extract_timestamp(file_name):
    match = re.search(r"_([0-9]{14})_", file_name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


def find_timestamp(path):
    match = re.search(r'(\d{8})(\d{6})_(\d{6})', path)
    start_time_str = f'{match.group(1)} {match.group(2)}'
    start_time = datetime.strptime(start_time_str, '%Y%m%d %H%M%S')
    start_time = start_time - timedelta(minutes=30)
    # print(f'Start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    time = pd.Timestamp(start_time)
    temb = time_encoding(time)
    return temb


def denormalize_matrix(normalized_matrix, min_val, max_val):

    denormalized_matrix = normalized_matrix * (max_val - min_val) + min_val
    return denormalized_matrix


def de_matrix(BT):
    # BT[0:1] = denormalize_matrix(BT[0:1], min_val=0, max_val=1.15)
    # BT[1:2] = denormalize_matrix(BT[1:2], min_val=0, max_val=1.15)
    # BT[2:3] = denormalize_matrix(BT[2:3], min_val=0, max_val=1.1)
    # BT[3:4] = denormalize_matrix(BT[3:4], min_val=0, max_val=1)
    # BT[4:5] = denormalize_matrix(BT[4:5], min_val=0, max_val=1.15)
    # BT[5:6] = denormalize_matrix(BT[5:6], min_val=0, max_val=1.15)
    # BT[6:7] = denormalize_matrix(BT[6:7], min_val=0, max_val=405)
    # BT[7:8] = denormalize_matrix(BT[7:8], min_val=0, max_val=341)
    BT[0:1] = denormalize_matrix(BT[0:1], min_val=150, max_val=310)
    BT[1:2] = denormalize_matrix(BT[1:2], min_val=160, max_val=310)
    BT[2:3] = denormalize_matrix(BT[2:3], min_val=150, max_val=330)
    BT[3:4] = denormalize_matrix(BT[3:4], min_val=170, max_val=340)
    BT[4:5] = denormalize_matrix(BT[4:5], min_val=170, max_val=350)
    BT[5:6] = denormalize_matrix(BT[5:6], min_val=170, max_val=350)
    BT[6:7] = denormalize_matrix(BT[6:7], min_val=170, max_val=330)
    # BT[15:16] = denormalize_matrix(BT[15:16], min_val=0, max_val=180)

    return BT

def read_txt2list(path):
    with open(path, "r") as f:
        l = f.readlines()
    return l


def validate_autoregressive_model(fixed_model, inference_frames, device, frame_history):
    fixed_model_0, fixed_model_1, fixed_model_2 = fixed_model
    torch.cuda.empty_cache()

    dataset_path = "FY4B_IR_brightness_temperature_dataset_path"

    datatime_list = glob.glob(os.path.join(dataset_path, "*"))
    datatime_list.sort()
    datatime_list = datatime_list[:]


    interval_count = 2
    interval_minutes = 30
    interval_timedelta = timedelta(minutes=interval_minutes)

    frames_list = []
    # for datatime in datatime_list:
    # frames_list = glob.glob(os.path.join(datatime_list, "*"))

    for datatime in datatime_list:
        frames = glob.glob(os.path.join(datatime, "*"))

        for frame in frames:
            frames_list.append(frame)

    frames_base_list = [os.path.basename(i) for i in frames_list]
    frames_base_list.sort(key=extract_timestamp)
    group = []
    groups= []
    prev_time = None


    for i in range(len(frames_base_list)):
        for file in frames_base_list[i:i + interval_count * interval_minutes // 15]:
            current_time = extract_timestamp(file)

            # if os.path.join(dir_path, re.search(r'\d{8}', file).group(), file)
            if prev_time is None or (current_time - prev_time) == interval_timedelta:
                next_file_path = os.path.join(dataset_path, re.search(r'\d{8}', file).group(), file)
                group.append(next_file_path)
                prev_time = current_time
                next_time = prev_time + interval_timedelta
                next_time_str = next_time.strftime('%Y%m%d%H%M%S')
                next_timestamp = next_time_str + "_" + (next_time + timedelta(minutes=14, seconds=59)).strftime(
                    '%Y%m%d%H%M%S')
                filename = file.replace(file[44:73], next_timestamp)

            if len(group) == interval_count:
                groups.append(group)
                # print(group, count)
                prev_time = None
                break
            else:
                if filename not in frames_base_list:
                    prev_time = None
                    break
        group = []


    groups = groups[:]
    predictions = []
    tembs = []
    tembs_tensors = []

    for index in range(len(groups)):
        group = groups[index]
        predictions = []

        for index in range(inference_frames):
            if index == 0:
                tembs_tensors.append(find_timestamp(group[index]).cuda(device).squeeze(1))
                tembs_tensors.append(find_timestamp(group[index+1]).cuda(device).squeeze(1))
                match = re.search(r'(\d{8})(\d{6})_(\d{6})', group[index+1])
                start_time_str = f'{match.group(1)} {match.group(2)}'
                start_time = datetime.strptime(start_time_str, '%Y%m%d %H%M%S') + timedelta(minutes=30)
                tembs_tensors.append(time_encoding(pd.Timestamp(start_time)).cuda(device).squeeze(1))
            elif index == 1:
                tembs_tensors.append(find_timestamp(group[1]).cuda(device).squeeze(1))
                match = re.search(r'(\d{8})(\d{6})_(\d{6})', group[1])
                start_time_str = f'{match.group(1)} {match.group(2)}'
                base_time = datetime.strptime(start_time_str, '%Y%m%d %H%M%S')
                tembs_tensors.append(time_encoding(pd.Timestamp(base_time + timedelta(minutes=30))).cuda(device).squeeze(1))
                tembs_tensors.append(time_encoding(pd.Timestamp(base_time + timedelta(minutes=60))).cuda(device).squeeze(1))
                base_time += timedelta(minutes=30)
            else:
                tembs_tensors.append(time_encoding(pd.Timestamp(base_time)).cuda(device).squeeze(1))
                tembs_tensors.append(time_encoding(pd.Timestamp(base_time + timedelta(minutes=30))).cuda(device).squeeze(1))
                tembs_tensors.append(time_encoding(pd.Timestamp(base_time + timedelta(minutes=60))).cuda(device).squeeze(1))
                base_time += timedelta(minutes=30)

            temb = torch.cat(tembs_tensors, dim=1)
            tembs_tensors = []
            tembs.append(temb)

        start_time = extract_timestamp(group[0])
        start_2_time = start_time + timedelta(minutes=30)
        start_2_time_str = start_2_time.strftime("%Y%m%d%H%M%S")
        save_path = f'/home/Data_Pool/zhurz/Fengyun4B-Pred/{start_2_time_str}'
        print(save_path)
        os.makedirs(save_path, exist_ok=True)

        first_timestamp = None
        second_timestamp = None
        stacked_timestamp = None
        output = None
        pred = None

        with torch.no_grad():
            for idx in range(inference_frames - frame_history):

                tembs[idx] =tembs[idx].float()
                if idx == 0:
                    first_timestamp = torch.from_numpy(np.load(group[idx])).cuda(device).squeeze()
                    second_timestamp = torch.from_numpy(np.load(group[idx + 1])).cuda(device).squeeze()
                    stacked_timestamp = torch.stack((first_timestamp, second_timestamp), dim=0)
                    stacked_timestamp = stacked_timestamp.float()
                    output = fixed_model_0(stacked_timestamp, tembs[idx]).squeeze()

                elif idx == 1:
                    first_timestamp = torch.from_numpy(np.load(group[idx])).cuda(device).squeeze()
                    second_timestamp = predictions[idx - 1].cuda(device).squeeze()
                    stacked_timestamp = torch.stack((first_timestamp, second_timestamp), dim=0)
                    stacked_timestamp = stacked_timestamp.float()
                    output = fixed_model_1(stacked_timestamp, tembs[idx]).squeeze()

                elif idx == 2:
                    first_timestamp = predictions[idx - 2].cuda(device).squeeze()
                    second_timestamp = predictions[idx - 1].cuda(device).squeeze()
                    stacked_timestamp = torch.stack((first_timestamp, second_timestamp), dim=0)
                    stacked_timestamp = stacked_timestamp.float()
                    output = fixed_model_2(stacked_timestamp, tembs[idx]).squeeze()

                elif idx >= 3:
                    first_timestamp = predictions[idx - 2].cuda(device).squeeze()
                    second_timestamp = predictions[idx - 1].cuda(device).squeeze()
                    stacked_timestamp = torch.stack((first_timestamp, second_timestamp), dim=0)
                    stacked_timestamp = stacked_timestamp.float()

                    output = fixed_model_2(stacked_timestamp, tembs[idx]).squeeze()

                predictions.append(output.cpu())
                pred = copy.deepcopy(output.cpu().squeeze())
                pred = de_matrix(pred)
                pred = np.asarray(pred)

                if idx == 1:
                    end_time_1h = start_time + timedelta(minutes=90)
                    current_time_str = end_time_1h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_1h + pd.Timedelta(seconds=59, minutes=14)).strftime("%Y%m%d%H%M%S")
                    np.save(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy', pred)
                    print(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')
                elif idx == 0:
                    end_time_05h = start_time + timedelta(minutes=60)
                    current_time_str = end_time_05h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_05h + pd.Timedelta(seconds=59, minutes=14)).strftime(
                        "%Y%m%d%H%M%S")
                    np.save(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy',
                        pred)
                    print(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')

                elif idx == 2:
                    end_time_15h = start_time + timedelta(minutes=120)
                    current_time_str = end_time_15h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_15h + pd.Timedelta(seconds=59, minutes=14)).strftime("%Y%m%d%H%M%S")
                    np.save(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy',
                        pred)
                    print(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')

                elif idx == 3:
                    end_time_2h =  start_time + timedelta(minutes=150)
                    current_time_str = end_time_2h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_2h + pd.Timedelta(seconds=59, minutes=14)).strftime("%Y%m%d%H%M%S")
                    np.save(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy', pred)
                    print(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')

                elif idx == 4:
                    end_time_25h = start_time + timedelta(minutes=180)
                    current_time_str = end_time_25h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_25h + pd.Timedelta(seconds=59, minutes=14)).strftime("%Y%m%d%H%M%S")
                    np.save(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy',
                        pred)
                    print(
                        f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')
                elif idx == 5:
                    end_time_3h = start_time + timedelta(minutes=210)
                    current_time_str = end_time_3h.strftime("%Y%m%d%H%M%S")
                    current_time_15min_str = (end_time_3h + pd.Timedelta(seconds=59, minutes=14)).strftime("%Y%m%d%H%M%S")
                    np.save(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy', pred)
                    print(f'{save_path}/FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_{current_time_str}_{current_time_15min_str}_4000M.npy')



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    device_number = 0

    device = "cuda:%d" % device_number
    model = build().to(device)

    fixed_model = copy.deepcopy(model)
    checkpoint_path = "path_to_model_0"

    checkpoint_fixed = torch.load(checkpoint_path, map_location="cuda:%d" % device_number)
    state_dict_fixed = checkpoint_fixed['state_dict']

    # 删除前缀'module.'
    for key in list(state_dict_fixed.keys()):
        state_dict_fixed[key[7:]] = state_dict_fixed.pop(key)
    fixed_model.load_state_dict(state_dict_fixed)
    fixed_model = fixed_model.eval()


    fixed_model_1 = copy.deepcopy(model)
    checkpoint_path_fixed = "path_to_model_1"

    checkpoint_fixed = torch.load(checkpoint_path_fixed, map_location="cuda:%d" % device_number)
    state_dict_fixed = checkpoint_fixed['state_dict']

    # 删除前缀'module.'
    for key in list(state_dict_fixed.keys()):
        state_dict_fixed[key[7:]] = state_dict_fixed.pop(key)
    fixed_model_1.load_state_dict(state_dict_fixed)
    fixed_model_1 = fixed_model_1.eval()

    fixed_model_2 = copy.deepcopy(model)
    checkpoint_path_fixed = "path_to_model_2"

    checkpoint_fixed = torch.load(checkpoint_path_fixed, map_location="cuda:%d" % device_number)
    state_dict_fixed = checkpoint_fixed['state_dict']

    # 删除前缀'module.'
    for key in list(state_dict_fixed.keys()):
        state_dict_fixed[key[7:]] = state_dict_fixed.pop(key)
    fixed_model_2.load_state_dict(state_dict_fixed)
    fixed_model_2 = fixed_model_2.eval()

    inference_frames = 10

    validate_autoregressive_model((fixed_model,fixed_model_1, fixed_model_2,
                                   ), inference_frames,device_number, 2)


