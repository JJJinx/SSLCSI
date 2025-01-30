import torch 
import torch.nn as nn
from mmcls.models.utils import Augments

from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class Classification_CSI_ResNet(BaseModel):
    """Simple CSI classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 head=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classification_CSI_ResNet, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)
    
    def extract_feat(self,csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if isinstance(csi,list):
            csi=csi[0]
        x = self.backbone(csi)  # tuple
        return x


    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]

        if self.augments is not None:
            csi, label = self.augments(csi, label)

        x = self.extract_feat(csi)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during test.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi = csi[0]
        x = self.extract_feat(csi)  # tuple
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

@ALGORITHMS.register_module()
class Classification_CSI_ResNet_Dual(BaseModel):
    """Simple CSI classification using both amplitude and phase.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 head=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classification_CSI_ResNet_Dual, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)
    
    def extract_feat(self,csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if isinstance(csi,list):
            csi=csi[0]
        amp,phase = self.backbone(csi)  # tuple
        return tuple([amp,phase])

    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]

        if self.augments is not None:
            csi, label = self.augments(csi, label)
        amp,phase = self.backbone(csi)
        #amp,phase = self.extract_feat(csi)
        outs = self.head(amp[0],phase[0])
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during test.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi = csi[0]
        amp,phase = self.extract_feat(csi)
        outs = self.head(amp[0],phase[0])
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))


@ALGORITHMS.register_module()
class Classification_CSI_ResNet_ARC(BaseModel):
    """Simple CSI classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 head=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classification_CSI_ResNet_ARC, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)
    
    def extract_feat(self,csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input CSI record of shape (N, A, C, T). (we can regard as an img of shape [NCHW])

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if isinstance(csi,list):
            csi=csi[0]
        batch = csi.size(0)
        num_ant = csi.size(1)
        out = []
        for ant_id in range(num_ant): 
            x = csi[:,ant_id,:,:].unsqueeze(1) #[B, 1, C ,T] #torch.Size([16, 1, 30, 500])
            x = self.backbone(x)[0] #[B, dim_out, C_out ,T_out] torch.Size([16, 512, 1, 16])
            out.append(x)
        out = torch.stack(out,1)# torch.Size([B, 3, dim_out, C_out ,T_out])
        out = out.view(out.size(0), out.size(1)*out.size(2),out.size(3),out.size(4)) #[B, 3*dim_out, C_out ,T_out]
        return tuple([out])


    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]

        if self.augments is not None:
            csi, label = self.augments(csi, label)

        x = self.extract_feat(csi)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during test.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi = csi[0]
        x = self.extract_feat(csi)  # tuple
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))


@ALGORITHMS.register_module()
class Classification_CSI_ResNet_ARC_Dual(BaseModel):
    """Simple CSI classification with ARC using both amp and phase.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 head=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classification_CSI_ResNet_ARC_Dual, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)
    
    def extract_feat(self,csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input CSI record of shape (N, A, C, T,2).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        batch = csi.size(0)
        num_ant = csi.size(1)
        amp_out = []
        phase_out = []
        for ant_id in range(num_ant):
            csi_ant = csi[:,ant_id,:,:,:].unsqueeze(1) #[B, 1, C ,T,2] #torch.Size([16, 1, 30, 500])
            amp,phase = self.backbone(csi_ant)   # output 2 lists
            amp_out.append(amp[0])
            phase_out.append(phase[0])
        amp_out = torch.stack(amp_out,1)# torch.Size([B, 3, dim_out, C_out ,T_out])
        phase_out = torch.stack(phase_out,1)

        amp_out = amp_out.view(amp_out.size(0), amp_out.size(1)*amp_out.size(2),amp_out.size(3),amp_out.size(4)) #[B, 3*dim_out, C_out ,T_out]
        phase_out = phase_out.view(phase_out.size(0), phase_out.size(1)*phase_out.size(2),phase_out.size(3),phase_out.size(4))
        return tuple([amp_out,phase_out])

    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]  #[N, 2, 30, 500, 2]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        amp,phase = self.extract_feat(csi)
        #amp,phase = self.head(amp,phase) 
        score = self.head(amp,phase) 
        loss_inputs = (score,label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during test.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi = csi[0]
        amp,phase = self.extract_feat(csi)
        #amp,phase = self.head(amp,phase) 
        score = self.head(amp,phase)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in score]  # NxC

        #amp_tensors = [out.cpu() for out in amp]  # NxC
        #phase_tensors =  [out.cpu() for out in phase]
        # out_tensors = []
        # for i in range(len(amp_tensors)):
        #     out_tensors.append((amp_tensors[i],phase_tensors[i]))
        return dict(zip(keys, out_tensors))