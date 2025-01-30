# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import math
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcls.utils import get_root_logger
from mmcls.models.utils import MultiheadAttention, resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone

from ..builder import BACKBONES
from ..utils import build_2d_sincos_position_embedding

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x


@BACKBONES.register_module()
class VisionTransformer_CSI_Dual(BaseBackbone):
    """Two tower Vision Transformer, one for amp and the other for phase.

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['mocov3-s', 'mocov3-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 1536,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['csi-s', 'csi-small'], {
                'embed_dims': 768,
                'num_layers': 6,
                'num_heads': 6,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['csi-as', 'csi-antsmall'], {
                'embed_dims': 384,
                'num_layers': 6,
                'num_heads': 6,
                'feedforward_channels': 1536
            }),    
    }
    
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 mask_ratio=0.75,
                 finetune=True,
                 stop_grad_conv1=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None):
        super(VisionTransformer_CSI_Dual, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        self.patch_size = patch_size
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.amp_patch_embed = PatchEmbed(**_patch_cfg)
        self.phase_patch_embed = PatchEmbed(**_patch_cfg)
        if stop_grad_conv1:
            self.amp_patch_embed.projection.weight.requires_grad = False
            self.amp_patch_embed.projection.bias.requires_grad = False
            self.phase_patch_embed.projection.weight.requires_grad = False
            self.phase_patch_embed.projection.bias.requires_grad = False


        self.patch_resolution = self.amp_patch_embed.init_out_size
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.norm_eval = norm_eval
        self.mask_ratio = mask_ratio
        self.finetune = finetune

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.amp_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))  # amplitude class token
        self.phase_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set position embedding
        # amp and phase share the same position embedding code
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.pos_embed.requires_grad = False
        self.frozen_stages = frozen_stages

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.amp_layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.amp_layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.phase_layers = ModuleList()
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.phase_layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm_amp_name, norm_amp = build_norm_layer(
                norm_cfg, self.embed_dims, postfix='amp') #return norm1_name is str; norm1 is nn.module
            self.add_module(self.norm_amp_name, norm_amp)
            self.norm_phase_name, norm_phase = build_norm_layer(
                norm_cfg, self.embed_dims, postfix='phase') 
            self.add_module(self.norm_phase_name, norm_phase)
        
        # whether freeze the backbone
        if not self.finetune: # when finetune is false, freeze the para
            self._freeze_stages()

    @property
    def norm_amp(self):
        return getattr(self, self.norm_amp_name)
    
    @property
    def norm_phase(self):
        return getattr(self, self.norm_phase_name)

    def init_weights(self):
        super(VisionTransformer_CSI_Dual, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            #trunc_normal_(self.pos_embed, std=0.02)
            # Use fixed 2D sin-cos position embedding
            pos_emb = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                cls_token=True)
            self.pos_embed.data.copy_(pos_emb)
            self.pos_embed.requires_grad = False

            # xavier_uniform initialization for PatchEmbed
            if isinstance(self.amp_patch_embed, PatchEmbed):
                val = math.sqrt(
                    6. / float(3 * reduce(mul, to_2tuple(self.patch_size), 1) +
                               self.embed_dims))
                nn.init.uniform_(self.amp_patch_embed.projection.weight, -val, val)
                nn.init.zeros_(self.amp_patch_embed.projection.bias)
            if isinstance(self.phase_patch_embed, PatchEmbed):
                val = math.sqrt(
                    6. / float(3 * reduce(mul, to_2tuple(self.patch_size), 1) +
                               self.embed_dims))
                nn.init.uniform_(self.phase_patch_embed.projection.weight, -val, val)
                nn.init.zeros_(self.phase_patch_embed.projection.bias)

            # initialization for linear layers
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if 'qkv' in name:
                        # treat the weights of Q, K, V separately
                        val = math.sqrt(
                            6. /
                            float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                        nn.init.uniform_(m.weight, -val, val)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.amp_cls_token, std=1e-6)
            nn.init.normal_(self.phase_cls_token, std=1e-6)


    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_stages(self):
        """Freeze params in backbone when linear probing."""
        for _, param in self.named_parameters():
            param.requires_grad = False

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmcv.utils import print_log
            logger = get_root_logger()
            print_log(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.',
                logger=logger)

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.amp_patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)


    def forward(self, x):
        '''
            x: input shape with [N,A,C,T,2(amp,ratio_ang)] 
        '''

        ####
        B = x.shape[0]
        ampx = x[:,:,:,:,0]          # [N A(1) C T]
        phasex = x[:,:,:,:,1]
        ### amp
        ampx, amp_patch_resolution = self.amp_patch_embed(ampx) #ampx shape[N,num_patches,embed_dim] [64, 150, 768]
        # append cls token
        amp_cls_tokens = self.amp_cls_token.expand(B, -1, -1)
        ampx = torch.cat((amp_cls_tokens, ampx), dim=1)
        ampx = ampx + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            amp_patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        ampx = self.drop_after_pos(ampx)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            ampx = ampx[:, 1:]

        amp_outs = []
        for i, layer in enumerate(self.amp_layers):
            ampx = layer(ampx)
            if i == len(self.amp_layers) - 1 and self.final_norm:
                ampx = self.norm_amp(ampx)
            
            if i in self.out_indices:
                B, _, C = ampx.shape
                if self.with_cls_token:
                    patch_token = ampx[:, 1:].reshape(B, *amp_patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = ampx[:, 0]
                else:
                    patch_token = ampx.reshape(B, *amp_patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                amp_outs.append(out)

        ### phase
        phasex, phase_patch_resolution = self.phase_patch_embed(phasex) #phasex shape[N,num_patches,embed_dim] [64, 150, 768]
        # append cls token
        phase_cls_tokens = self.phase_cls_token.expand(B, -1, -1)
        phasex = torch.cat((phase_cls_tokens, phasex), dim=1)
        phasex = phasex + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            phase_patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        phasex = self.drop_after_pos(phasex)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            phasex = phasex[:, 1:]

        phase_outs = []
        for i, layer in enumerate(self.phase_layers):
            phasex = layer(phasex)
            if i == len(self.phase_layers) - 1 and self.final_norm:
                phasex = self.norm_phase(phasex)
            
            if i in self.out_indices:
                B, _, C = phasex.shape
                if self.with_cls_token:
                    patch_token = phasex[:, 1:].reshape(B, *phase_patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = phasex[:, 0]
                else:
                    patch_token = phasex.reshape(B, *phase_patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                phase_outs.append(out)
        # tuple(out)
        return tuple([amp_outs,phase_outs])
               

    def train(self, mode=True):
        super(VisionTransformer_CSI_Dual, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

