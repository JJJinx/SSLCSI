import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class RelativeLocNeck_CSI(BaseModule):
    """The neck of relative patch location: fc-bn-relu-dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN1d').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True,
                 norm_cfg=dict(type='BN1d'),
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RelativeLocNeck_CSI, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(in_channels * 2, out_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = build_norm_layer(
            dict(**norm_cfg, momentum=0.003), out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return [x]
