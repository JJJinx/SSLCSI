import torch
from mmcls.models import VisionTransformer
from mmcls.models.backbones.base_backbone import BaseBackbone
from torch import nn

from ..builder import BACKBONES
from ..utils import build_2d_sincos_position_embedding


@BACKBONES.register_module()
class MAEViT_CSI(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
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
    
    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 mask_ratio=0.75,
                 finetune=True,
                 init_cfg=None):
        super(MAEViT_CSI,self).__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.finetune = finetune
        if not self.finetune: # when finetune is false, freeze the para
            self._freeze_stages()

    def init_weights(self):
        super(MAEViT_CSI, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                self.patch_resolution,
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            w = self.patch_embed.projection.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.cls_token, std=.02)

            self.apply(self._init_weights)

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

    def random_masking(self, x, mask_ratio=0.75):
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.tensor): Image with data augmentation applied.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            tuple[Tensor, Tensor, Tensor]: masked image, mask and the ids
                to restore original image.

            - x_masked (Tensor): masked image.
            - mask (Tensor): mask used to mask image.
            - ids_restore (Tensor): ids to restore original image.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        """
            x: input shape with [N,A,C,T]
        """
        B = x.shape[0]
        # self.patch_embed is the PatchEmbed function in https://github.com/open-mmlab/mmcv/blob/1.x/mmcv/cnn/bricks/transformer.py
        #In particular, it involves feed x to a convolutional network to obtain a feature with shape [N, embed_dim, out_C, out_T];
        # then after transposition, the dim C and T is multiplied to obtain [N, out_C * out_T, embed_dim] = [N, num_patches, embed_dim].
        x = self.patch_embed(x)[0] #[N,num_patches,embed_dim]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio) # x shape [N,num_patches*(1-mask_ratio),embed_dim] 
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        # x shape [N,150,embeddim]
        x = torch.cat((cls_tokens, x), dim=1)# x shape [N,num_patches*(1-mask_ratio)+1,embed_dim] 
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return (x, mask, ids_restore)

