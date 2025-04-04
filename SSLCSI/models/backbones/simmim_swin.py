# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcls.models import SwinTransformer
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import BACKBONES


@BACKBONES.register_module()
class SimMIMSwinTransformer(SwinTransformer):
    """Swin Transformer for SimMIM.

    Args:
        Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate after embedding.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        out_indices (tuple): Layers to be outputted. Defaults to (3, ).
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer at end
            of backone. Defaults to dict(type='LN')
        stage_cfgs (Sequence | dict): Extra config dict for each
            stage. Defaults to empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to empty dict.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'T',
                 img_size: Union[Tuple[int, int], int] = 224,
                 in_channels: int = 3,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 out_indices: tuple = (3, ),
                 use_abs_pos_embed: bool = False,
                 with_cp: bool = False,
                 frozen_stages: bool = -1,
                 norm_eval: bool = False,
                 norm_cfg: dict = dict(type='LN'),
                 stage_cfgs: Union[Sequence, dict] = dict(),
                 patch_cfg: dict = dict(),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            in_channels=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            use_abs_pos_embed=use_abs_pos_embed,
            with_cp=with_cp,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
            norm_cfg=norm_cfg,
            stage_cfgs=stage_cfgs,
            patch_cfg=patch_cfg,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize weights."""
        super(SwinTransformer, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        trunc_normal_(self.mask_token, mean=0, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Sequence[torch.Tensor]:
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        them.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor): Masks used to construct masked images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        x, hw_shape = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed

        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1,
                                                           2).contiguous()
                outs.append(out)

        return tuple(outs)
