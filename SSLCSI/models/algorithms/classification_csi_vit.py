import torch 
import torch.nn as nn
from mmcls.models.utils import Augments
import numpy as np

from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class Classification_CSI_VIT(BaseModel):
    """Simple CSI classification when using ViT backbone.

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
        super(Classification_CSI_VIT, self).__init__(init_cfg)
        self.with_sobel = False
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, csi):
        """Function to extract features from backbone. For TSNE or CKA.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).

        Returns:
            tensor: backbone outputs.
        """
        csi = csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        x = self.backbone(csi)
        x = tuple([x[0][-1]])
        return x # N C T
    
    def extract_feat_forward(self, csi):
        x = self.backbone(csi) # shape [N, embed_dim] [64, 768]
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
        csi=csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        # only the class token is used for classification
        x = self.extract_feat_forward(csi)  # list (patch_token,cls_token)
        # selecct the cls_token
        x = x[0][-1] 
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
        csi=csi[0]
        # only the class token is used for classification
        x = self.extract_feat_forward(csi)
        x = x[0][-1] 
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))


@ALGORITHMS.register_module()
class Classification_CSI_VIT_ARC(BaseModel):
    """Simple CSI classification with ARC based on ViT.

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
        super(Classification_CSI_VIT_ARC, self).__init__(init_cfg)
        self.with_sobel = False
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, csi):
        """Function to extract features from backbone. For TSNE or CKA.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).

        Returns:
            tensor: backbone outputs.
        """
        csi = csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        batch = csi.size(0)
        num_ant = csi.size(1)
        outs = []
        #numbers = np.arange(csi.size(1))
        #rand_ant1,rand_ant2 = numbers[:2],numbers[2:]
        for ant_id in range(num_ant):
        #for ant_id in (rand_ant1,rand_ant2):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            if len(x.shape)>4:
                x = x.squeeze()
            x = self.backbone(x)[0][-1] # shape [N,embed_dim] 
            outs.append(x)
        outs = torch.stack(outs,1) #N,A,embed_dim [64, 3, 768]
        outs = outs.view(outs.size(0), -1) #[B,3*outdim]
        return tuple([outs]) # N C T
    

    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi=csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        batch = csi.size(0)
        num_ant = csi.size(1)
        outs = []
        #numbers = np.arange(csi.size(1))
        #rand_ant1,rand_ant2 = numbers[:2],numbers[2:]
        for ant_id in range(num_ant):
        #for ant_id in (rand_ant1,rand_ant2):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            if len(x.shape)>4:
                x = x.squeeze()
            x = self.backbone(x)[0][-1] # shape [N,embed_dim] 
            outs.append(x)
        outs = torch.stack(outs,1) #N,A,embed_dim [64, 3, 768]
        outs = outs.view(outs.size(0), -1) #[B,3*outdim]
        # only the class token is used for classification
        x = outs 
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
        csi=csi[0]

        batch = csi.size(0)
        num_ant = csi.size(1)
        outs = []
        #numbers = np.arange(csi.size(1))
        #rand_ant1,rand_ant2 = numbers[:2],numbers[2:]
        for ant_id in range(num_ant):
        #for ant_id in (rand_ant1,rand_ant2):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            if len(x.shape)>4:
                x = x.squeeze()
            x = self.backbone(x)[0][-1] # shape [N,embed_dim] 
            outs.append(x)
        outs = torch.stack(outs,1) #N,A,embed_dim [64, 3, 768]
        outs = outs.view(outs.size(0), -1) #[B,3*outdim]
        # only the class token is used for classification
        x = outs
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))



@ALGORITHMS.register_module()
class Classification_CSI_VIT_Dual(BaseModel):
    """Simple CSI classification using both amp and phase based on ViT.

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
        super(Classification_CSI_VIT_Dual, self).__init__(init_cfg)
        self.with_sobel = False
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, csi):
        """Function to extract features from backbone. For TSNE or CKA.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).

        Returns:
            tensor: backbone outputs.
        """
        csi = csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        batch = csi.size(0)
        amp_outs,phase_outs = self.backbone(csi)        

        return tuple([amp_outs,phase_outs]) # N C T
    

    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi=csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)

        batch = csi.size(0)
        amp_outs,phase_outs = self.backbone(csi)  
        
        # only the class token is used for classification
        outs = self.head(amp_outs[0][-1],phase_outs[0][-1]) # return the sum of logits of amp and phase
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
        csi=csi[0]

        batch = csi.size(0)
        amp_outs,phase_outs = self.backbone(csi)  
        # only the class token is used for classification
        outs = self.head(amp_outs[0][-1],phase_outs[0][-1]) # return the sum of logits of amp and phase
        
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))



@ALGORITHMS.register_module()
class Classification_CSI_VIT_ARC_Dual(BaseModel):
    """Simple CSI classification based on ViT with ARC using both amp and phase.

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
        super(Classification_CSI_VIT_ARC_Dual, self).__init__(init_cfg)
        self.with_sobel = False
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, csi):
        """Function to extract features from backbone. For TSNE or CKA.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).

        Returns:
            tensor: backbone outputs.
        """
        csi = csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)
        batch = csi.size(0)
        num_ant = csi.size(1)
        amp_outs = []
        phase_outs = []
        for ant_id in range(num_ant):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            ampx,phasex = self.backbone(x)
            amp_outs.append(ampx[0][-1]) # shape [N,embed_dim] 
            phase_outs.append(phasex[0][-1])
        amp_outs = torch.stack(amp_outs,1) #N,A,embed_dim [64, 2, 768]
        phase_outs = torch.stack(phase_outs,1) #N,A,embed_dim [64, 2, 768]
        amp_outs = amp_outs.view(amp_outs.size(0), -1) #[B,2*outdim]
        phase_outs = phase_outs.view(phase_outs.size(0), -1) #[B,2*outdim]

        return tuple([amp_outs,phase_outs]) # N C T
    

    def forward_train(self, csi, label, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): Input CSI record of shape (N, A, C, T,2).
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi=csi[0]
        if self.augments is not None:
            csi, label = self.augments(csi, label)

        batch = csi.size(0)
        num_ant = csi.size(1)
        amp_outs = []
        phase_outs = []
        for ant_id in range(num_ant):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            ampx,phasex = self.backbone(x)
            amp_outs.append(ampx[0][-1]) # shape [N,embed_dim] 
            phase_outs.append(phasex[0][-1])
        amp_outs = torch.stack(amp_outs,1) #N,A,embed_dim [64, 2, 768]
        phase_outs = torch.stack(phase_outs,1) #N,A,embed_dim [64, 2, 768]
        amp_outs = amp_outs.view(amp_outs.size(0), -1) #[B,2*outdim]
        phase_outs = phase_outs.view(phase_outs.size(0), -1) #[B,2*outdim]

        # only the class token is used for classification
        outs = self.head(amp_outs,phase_outs) # return the sum of logits of amp and phase
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
        csi=csi[0]

        batch = csi.size(0)
        num_ant = csi.size(1)
        amp_outs = []
        phase_outs = []
        for ant_id in range(num_ant):
            x = csi[:,ant_id,:,:].unsqueeze(1) #B 1 C T
            ampx,phasex = self.backbone(x)
            amp_outs.append(ampx[0][-1]) # shape [N,embed_dim] 
            phase_outs.append(phasex[0][-1])
        amp_outs = torch.stack(amp_outs,1) #N,A,embed_dim [64, 3, 768]
        phase_outs = torch.stack(phase_outs,1) #N,A,embed_dim [64, 3, 768]
        amp_outs = amp_outs.view(amp_outs.size(0), -1) #[B,3*outdim]
        phase_outs = phase_outs.view(phase_outs.size(0), -1) #[B,3*outdim]

        # only the class token is used for classification
        outs = self.head(amp_outs,phase_outs) # return the sum of logits of amp and phase
        
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))
