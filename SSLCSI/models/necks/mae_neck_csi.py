import torch
import torch.nn as nn
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import build_2d_sincos_position_embedding


@NECKS.register_module()
class MAEPretrainDecoder_CSI(BaseModule):
    """Decoder for MAE Pre-training.

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.

    Some of the code is borrowed from
    `https://github.com/facebookresearch/mae`.

    Example:
        >>> from mmselfsup.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches=196,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 data_mode='OR',
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(MAEPretrainDecoder_CSI, self).__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        if isinstance(patch_size,int):
            self.patch_size=(patch_size,patch_size)
        else:
            self.patch_size = patch_size

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size[0]*self.patch_size[1] * in_chans, bias=True)
        self.data_mode = data_mode

    def init_weights(self):
        super(MAEPretrainDecoder_CSI, self).init_weights()
        if self.data_mode == 'OR': # wifi office
            pos_emb_tuple=(int(30//self.patch_size[0]),int(2000//self.patch_size[1]))
        if self.data_mode == 'WIDAR': # WIDAR
            pos_emb_tuple=(int(30//self.patch_size[0]),int(500//self.patch_size[1]))
        if self.data_mode == 'WIDAR_all': # WIDAR
            pos_emb_tuple=(int(60//self.patch_size[0]),int(500//self.patch_size[1]))
        if self.data_mode == 'Signfi': # Signfi
            pos_emb_tuple=(int(30//self.patch_size[0]),int(200//self.patch_size[1]))
        if self.data_mode == 'Falldefi': # Falldefi
            pos_emb_tuple=(int(30//self.patch_size[0]),int(10000//self.patch_size[1]))
        if self.data_mode == 'Falldefi_all': # Falldefi_all
            pos_emb_tuple=(int(60//self.patch_size[0]),int(10000//self.patch_size[1]))
        if self.data_mode == 'HUAWEI': # Huawei
            pos_emb_tuple=(int(50//self.patch_size[0]),int(90//self.patch_size[1]))
        if self.data_mode == 'CSIDA': # CSIDA
            pos_emb_tuple=(int(114//self.patch_size[0]),int(1800//self.patch_size[1]))
        if self.data_mode == 'CSIDA_all': # CSIDA
            pos_emb_tuple=(int(228//self.patch_size[0]),int(1800//self.patch_size[1]))
        # initialize position embedding of MAE decoder
        decoder_pos_embed = build_2d_sincos_position_embedding(
            pos_emb_tuple,
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def decoder_norm(self):
        return getattr(self, self.decoder_norm_name)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) #[N, patch_num, 512]
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) #[N, patch_num, 512]

        x = torch.cat([x[:, :1, :], x_], dim=1) #[N, patch_num+1, 512] 
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x) #[N, 151, 512]

        # predictor projection
        x = self.decoder_pred(x) #[N, 151, 300]

        # remove cls token
        x = x[:, 1:, :] #[N, patch_num, 300]
        return [x]
