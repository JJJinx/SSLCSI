import torch.nn as nn
import torch
import torchvision
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import HEADS
from ..utils import accuracy
from ..utils.csi_utils.loss_for_long_tail import FocalLoss,balanced_softmax_loss

@HEADS.register_module()
class ClsCsiHead(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 focal_loss_flag=False,
                 ):
        super(ClsCsiHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.focal_loss_flag = focal_loss_flag
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_focal = FocalLoss()

        if self.with_avg_pool:
            self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, T).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]

        if self.with_avg_pool:
            assert x.dim() <= 4, \
                f'Tensor must has dims no bigger than 4, got: {x.dim()}.'
            if x.dim() == 4:
                x = self.avg_pool2d(x)
            if x.dim() == 3:
                x = self.avg_pool1d(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        if self.focal_loss_flag:
            losses['loss'] = self.criterion_focal(cls_score[0], labels)
        else:
            losses['loss'] = self.criterion(cls_score[0], labels)
        losses['acc'] = accuracy(cls_score[0], labels)
        return losses


@HEADS.register_module()
class ClsCsiHead_Dual(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(ClsCsiHead_Dual, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.Softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        
        if self.with_avg_pool:
            self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls_amp = nn.Linear(in_channels, num_classes)
        self.fc_cls_phase = nn.Linear(in_channels, num_classes)

    def forward(self,amp,phase):
        """Forward head.

        Args:
            amp ([Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, T).

        Returns:
            list[Tensor]: A list of class scores.
        """        
        if self.with_avg_pool:
            assert amp.dim() <= 4, \
                f'Tensor must has dims no bigger than 4, got: {x.dim()}.'
            if amp.dim() == 4:
                amp = self.avg_pool2d(amp)
                phase = self.avg_pool2d(phase)
            if amp.dim() == 3:
                amp = self.avg_pool1d(amp)
                phase = self.avg_pool1d(phase)
        amp = amp.view(amp.size(0), -1)
        phase = phase.view(phase.size(0), -1)
        amp_score = self.fc_cls_amp(amp)
        phase_score = self.fc_cls_phase(phase)
        amp_logit = self.Softmax(amp_score)
        phase_logit = self.Softmax(phase_score)
        cls_score = torch.exp(0.5*(amp_logit+phase_logit))
        return [cls_score]

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        # amp_logit = self.Softmax(amp_score)
        # phase_logit = self.Softmax(phase_score)
        # cls_score = torch.exp(0.5*(amp_logit+phase_logit))
        losses['loss'] = self.criterion(torch.log(cls_score[0]), labels)
        losses['acc'] = accuracy(torch.log(cls_score[0]), labels)
        return losses


@HEADS.register_module()
class MultiCSIClsHead(BaseModule):
    """Multiple classifier heads.
        fc-bn-relu-fc
    """
    def __init__(self,
                 in_channels=320,
                 hid_channels=128,
                 num_classes=22,
                 with_avg_pool=False,
                 focal_loss_flag=False,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MultiCSIClsHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.head = nn.Linear(hid_channels, num_classes)
        self.bn = nn.BatchNorm1d(hid_channels, affine=False, eps=1e-6)
        self.focal_loss_flag = focal_loss_flag
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_focal = FocalLoss()
        self.relu = nn.ReLU(inplace=True)
        if self.with_avg_pool:
            self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.fc0.bias, 0)
        trunc_normal_(self.fc0.weight, std=0.01)
        
    def forward(self, x):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, T).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            assert x.dim() <= 4, \
                f'Tensor must has dims no bigger than 4, got: {x.dim()}.'
            if x.dim() == 4:
                x = self.avg_pool2d(x)
            if x.dim() == 3:
                x = self.avg_pool1d(x)
            
        x = x.view(x.size(0), -1) # shape [in_chanel,1]
        x = self.fc0(x)
        x = self.bn(x)
        x = self.relu(x)
        cls_score = self.head(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        for i, s in enumerate(cls_score):
            # keys must contain "loss"
            if self.focal_loss_flag:
                losses[f'loss.{i + 1}'] = self.criterion_focal(s, labels)
            else:
                losses[f'loss.{i + 1}'] = self.criterion(s, labels)
            losses[f'acc.{i + 1}'] = accuracy(s, labels)
        return losses


@HEADS.register_module()
class MultiCSIClsHead_Dual(BaseModule):
    """Multiple classifier heads.

    """
    def __init__(self,
                 in_channels=320,
                 hid_channels=128,
                 num_classes=22,
                 with_avg_pool=False,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MultiCSIClsHead_Dual, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.amp_fc0 = nn.Linear(in_channels, hid_channels)
        self.amp_head = nn.Linear(hid_channels, num_classes)
        self.amp_bn = nn.BatchNorm1d(hid_channels, affine=False, eps=1e-6)
        self.phase_fc0 = nn.Linear(in_channels, hid_channels)
        self.phase_head = nn.Linear(hid_channels, num_classes)
        self.phase_bn = nn.BatchNorm1d(hid_channels, affine=False, eps=1e-6)

        self.Softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

        self.relu = nn.ReLU(inplace=True)
        if self.with_avg_pool:
            self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        nn.init.constant_(self.amp_head.bias, 0)
        trunc_normal_(self.amp_head.weight, std=0.01)
        nn.init.constant_(self.amp_fc0.bias, 0)
        trunc_normal_(self.amp_fc0.weight, std=0.01)

        nn.init.constant_(self.phase_head.bias, 0)
        trunc_normal_(self.phase_head.weight, std=0.01)
        nn.init.constant_(self.phase_fc0.bias, 0)
        trunc_normal_(self.phase_fc0.weight, std=0.01)
        
    def forward(self,amp,phase):
        """Forward head.

        Args:
            amp (Tensor): Feature maps of backbone,
                each tensor has shape (N, C, T).

        Returns:
            list[Tensor]: A list of class scores.
        """
        if self.with_avg_pool:
            assert amp.dim() <= 4, \
                f'Tensor must has dims no bigger than 4, got: {amp.dim()}.'
            if amp.dim() == 4:
                amp = self.avg_pool2d(amp)
                phase = self.avg_pool2d(phase)
            if amp.dim() == 3:
                amp = self.avg_pool1d(amp)
                phase = self.avg_pool1d(phase)
            
        amp = amp.view(amp.size(0), -1) # shape [in_chanel,1]
        phase = phase.view(phase.size(0), -1) 

        amp = self.amp_fc0(amp)
        phase = self.phase_fc0(phase)
        amp = self.amp_bn(amp)
        phase = self.phase_bn(phase)
        amp = self.relu(amp)
        phase = self.relu(phase)
        amp_score = self.amp_head(amp)
        phase_score = self.phase_head(phase)
        amp_logit = self.Softmax(amp_score)
        phase_logit = self.Softmax(phase_score)
        cls_score = torch.exp(0.5*(amp_logit+phase_logit))
        return [cls_score]

    def loss(self,cls_score, labels):
        """Compute the loss."""
        losses = dict()
        # amp_logit = self.Softmax(amp_score)
        # phase_logit = self.Softmax(phase_score)
        # cls_score = torch.exp(0.5*(amp_logit+phase_logit))
        for i, s in enumerate(cls_score):
            # keys must contain "loss"
            losses[f'loss.{i + 1}'] = self.criterion(torch.log(s), labels)
            losses[f'acc.{i + 1}'] = accuracy(torch.log(s), labels)
        return losses
