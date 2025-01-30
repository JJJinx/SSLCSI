import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MoCoV2Neck_CSI(BaseModule):
    """The non-linear neck of MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 init_cfg=None):
        super(MoCoV2Neck_CSI, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
            self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() ==4:
                x = self.avgpool2d(x)
            if x.dim() ==3:
                x = self.avgpool1d(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module()
class MoCoV2Neck_CSI_Dual(BaseModule):
    """The non-linear neck of MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 init_cfg=None):
        super(MoCoV2Neck_CSI_Dual, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.amp_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
            self.amp_avgpool1d = nn.AdaptiveAvgPool1d(1)
            self.phase_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
            self.phase_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.amp_mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
        self.phase_mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
    
    def forward(self,amp,phase):
        #assert len(x) == 1
        amp = amp[0]
        phase = phase[0]
        if self.with_avg_pool:
            if amp.dim() ==4:
                amp = self.amp_avgpool2d(amp)
                phase = self.phase_avgpool2d(phase)
            if amp.dim() ==3:
                amp = self.amp_avgpool1d(amp)
                phase = self.phase_avgpool1d(phase)
        amp_out = self.amp_mlp(amp.view(amp.size(0), -1))
        phase_out = self.phase_mlp(phase.view(phase.size(0), -1))
        return [amp_out,phase_out]