import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class NonLinearCsiNeck_Dual(BaseModule):
    """The non-linear neck.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 vit_backbone=False,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=[
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(NonLinearCsiNeck_Dual, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.vit_backbone = vit_backbone
        if with_avg_pool:
            self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
            self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.amp_fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.amp_bn0 = build_norm_layer(norm_cfg, hid_channels)[1]
        self.phase_fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.phase_bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.amp_fc_names = []
        self.amp_bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'amp_fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'amp_bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.amp_bn_names.append(f'amp_bn{i}')
            else:
                self.add_module(
                    f'amp_fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'amp_bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.amp_bn_names.append(f'amp_bn{i}')
                else:
                    self.amp_bn_names.append(None)
            self.amp_fc_names.append(f'amp_fc{i}')
        self.phase_fc_names = []
        self.phase_bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'phase_fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'phase_bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.phase_bn_names.append(f'phase_bn{i}')
            else:
                self.add_module(
                    f'phase_fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'phase_bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.phase_bn_names.append(f'phase_bn{i}')
                else:
                    self.phase_bn_names.append(None)
            self.phase_fc_names.append(f'phase_fc{i}')

    def forward(self, amp ,phase):
        #assert len(x) == 1
        amp = amp[0]
        phase = phase[0]
        if self.vit_backbone:
            amp = amp[-1]
            phase = phase[-1]
        if self.with_avg_pool:
            if amp.dim() ==4:
                amp = self.avgpool2d(amp)
                phase = self.avgpool2d(phase)
            if amp.dim() ==3:
                amp = self.avgpool1d(amp)
                phase = self.avgpool1d(phase)
        
        amp = amp.view(amp.size(0), -1)
        phase = phase.view(phase.size(0), -1)
        amp = self.amp_fc0(amp)
        amp = self.amp_bn0(amp)
        phase = self.phase_fc0(phase)
        phase = self.phase_bn0(phase)

        for fc_name, bn_name in zip(self.amp_fc_names, self.amp_bn_names):
            fc = getattr(self, fc_name)
            amp = self.relu(amp)
            amp = fc(amp)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                amp = bn(amp)
        for fc_name, bn_name in zip(self.phase_fc_names, self.phase_bn_names):
            fc = getattr(self, fc_name)
            phase = self.relu(phase)
            phase = fc(phase)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                phase = bn(phase)
        return [amp,phase]



@NECKS.register_module()
class TCnonlinearNeck(BaseModule):
    """The non-linear neck from both time and channel axis.

    Structure: fc(channel)-bn-fc(temporal)-bn-reshape-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=320,
                 in_temporal=500,
                 hid_channels=64,
                 out_channels=128,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 vit_backbone=False,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=[
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(TCnonlinearNeck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.vit_backbone = vit_backbone
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc_channel = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn_channel = build_norm_layer(norm_cfg, hid_channels)[1]
        self.fc_temporal = nn.Linear(in_temporal, hid_channels, bias=with_bias)
        self.bn_temporal = build_norm_layer(norm_cfg, hid_channels)[1]
        self.fc = nn.Linear(hid_channels**2, out_channels, bias=with_bias)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        # input shape [num_view*N, T, dim ]
        x = self.fc_channel(x)
        x = x.transpose(1,2) #shape [num_view*N, dim, T ]
        x = self.fc_temporal(x)
        x = x.view(x.size(0), -1)  #shape [num_view*N, dim**2 ]
        x = self.relu(x)
        x = self.fc(x)
        x = self.bn(x)
        return [x]
