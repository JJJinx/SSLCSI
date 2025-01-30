from pickle import TRUE
from sklearn.neighbors import radius_neighbors_graph
import torch
import torch.nn as nn
import numpy as np

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import GatherLayer
from .base import BaseModel


@ALGORITHMS.register_module()
class SimCLR_CSI(BaseModel):
    """SimCLR for CSI data.

    Implementation of `A Simple Framework for Contrastive Learning
    of Visual Representations <https://arxiv.org/abs/2002.05709>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None,AE_model= False):
        super(SimCLR_CSI, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        self.recon_cretirion = nn.MSELoss()

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda() #Diagonal elements are 0, others are 1. Mask diagonal.
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda() #2N-1
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): A list of input CSI with shape (N, C, T) or (N,A,C,T)
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(csi)
        return x

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input CSI with shape
                (N, C, T) or (NACT). Length of list equals to num_views (2)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        csi = torch.stack(csi, 1) 
        if 'ResNet' not in self.backbone.__class__.__name__:
            ## csi shape [N,2,C,T]
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3))) # shape [num_view*N, C, T ] 

        else:
            ## csi shape [N,2,A,C,T]
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape  [2N,A, C, T]
            #Using A and B as two views, after reshaping ,the tensor turns to [A,A,...,A,B,B,...,B]
        x = self.extract_feat(csi)  # 2n x hiddim
        z = self.neck(x)[0]  # (2n) x dim
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10) #regularization, get the direction information of feature z
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N) x dim
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1) 
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1) 
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)

        return losses

@ALGORITHMS.register_module()
class SimCLR_CSI_Dual(BaseModel):
    """SimCLR for CSI data using both amp and phase.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None,AE_model= False):
        super(SimCLR_CSI_Dual, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        self.recon_cretirion = nn.MSELoss()

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda() 
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda() #2N-1 
        neg_mask[pos_ind] = 0 
        return mask, pos_ind, neg_mask

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input CSI record of shape (N, C, T,2) or (N, A, C, T, 2).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        amp_outs,phase_outs = self.backbone(csi)
        return amp_outs,phase_outs

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input CSI with shape (N, C, T,2) or (N, A, C, T, 2). 
            Length of list equals to num_views (2)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        csi = torch.stack(csi, 1)  # (N,2,A, C, T,2)
        if 'ResNet' not in self.backbone.__class__.__name__:
            ## csi shape [N,2,C,T,2]
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape [num_view*N, C, T ] 

        else:
            ## csi shape [N,2,A,C,T,2]
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4), csi.size(5))) # shape  [2N,A, C, T]
        amp,phase = self.extract_feat(csi)  # 2n x hiddim
        z_amp,z_phase = self.neck(amp,phase)  # (2n) x dim

        z_amp = z_amp / (torch.norm(z_amp, p=2, dim=1, keepdim=True) + 1e-10) 
        z_amp = torch.cat(GatherLayer.apply(z_amp), dim=0)  # (2N) x dim
        assert z_amp.size(0) % 2 == 0
        z_phase = z_phase / (torch.norm(z_phase, p=2, dim=1, keepdim=True) + 1e-10) 
        z_phase = torch.cat(GatherLayer.apply(z_phase), dim=0)  # (2N) x dim
        assert z_phase.size(0) % 2 == 0
        N = z_amp.size(0) // 2
        s_amp = torch.matmul(z_amp, z_amp.permute(1, 0))  # (2N)x(2N)
        s_phase = torch.matmul(z_phase, z_phase.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1) 
        s_amp = torch.masked_select(s_amp, mask == 1).reshape(s_amp.size(0), -1) 
        positive_amp = s_amp[pos_ind].unsqueeze(1)  # (2N)x1
        s_phase = torch.masked_select(s_phase, mask == 1).reshape(s_phase.size(0), -1) 
        positive_phase = s_phase[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative_amp = torch.masked_select(s_amp, neg_mask == 1).reshape(s_amp.size(0), -1)
        negative_phase = torch.masked_select(s_phase, neg_mask == 1).reshape(s_phase.size(0), -1)
        losses = self.head(positive_amp,positive_phase, negative_amp,negative_phase)
        return losses



@ALGORITHMS.register_module()
class SimCLR_CSI_ARC(BaseModel):
    """SimCLR for CSI data with ARC.
    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(SimCLR_CSI_ARC, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda() 
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda() #2N-1
        neg_mask[pos_ind] = 0 
        return mask, pos_ind, neg_mask

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input csi record of shape (N,A ,C,T).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(csi)
        return x

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input CSI data with shape
                (N,A ,C,T). List length equal to num_view (2)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=3, size=2)
        csi_ant1 = csi[0][:,rand_ant1,:,:]
        csi_ant2 = csi[1][:,rand_ant2,:,:] # shape [N, C, T] [32, 30, 500]
        csi = torch.stack([csi_ant1,csi_ant2], 1) # [N, 2, C, T] [32, 2, 30, 500]
        if 'ResNet' not in self.backbone.__class__.__name__:
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3))) # shape [num_view*N, C, T ] 

        else:
            csi = csi.unsqueeze(2)
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape  [2N, 1, C, T]
        x = self.extract_feat(csi)  # list [2N,512(num_outchannel),C_out,T_out] that is [64, 512, 1, 16]
        z = self.neck(x)[0]  # (2n) x dim
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10) 
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N) x dim
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1) 
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1) 
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses

@ALGORITHMS.register_module()
class SimCLR_CSI_ARC_Dual(BaseModel):
    """SimCLR for CSI data with ARC using both amp and phase.


    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(SimCLR_CSI_ARC_Dual, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda() 
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda() #2N-1
        neg_mask[pos_ind] = 0 
        return mask, pos_ind, neg_mask

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input csi record of shape (N,A ,C,T,2).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        amp_outs,phase_outs = self.backbone(csi)
        return amp_outs,phase_outs

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input CSI data with shape
                (N,A ,C,T,2). List length equal to num_view (2)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi.size(dim=1), size=2)
        csi_ant1 = csi[0][:,rand_ant1,:,:]
        csi_ant2 = csi[1][:,rand_ant2,:,:] # shape [N, C, T,2] [32, 30, 500,2]
        csi = torch.stack([csi_ant1,csi_ant2], 1) # [N, 2, C, T,2] [32, 2, 30, 500,2]
        if 'ResNet' not in self.backbone.__class__.__name__:
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape [num_view*N, C, T,2] 
        else:
            csi = csi.unsqueeze(2)
            csi = csi.reshape(
            (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape  [2N, 1, C, T,2]
        amp,phase = self.extract_feat(csi)  # list [2N,512(num_outchannel),C_out,T_out] that is [64, 512, 1, 16]
        z_amp,z_phase = self.neck(amp,phase)  # (2n) x dim
        z_amp = z_amp / (torch.norm(z_amp, p=2, dim=1, keepdim=True) + 1e-10) 
        z_amp = torch.cat(GatherLayer.apply(z_amp), dim=0)  # (2N) x dim
        assert z_amp.size(0) % 2 == 0
        z_phase = z_phase / (torch.norm(z_phase, p=2, dim=1, keepdim=True) + 1e-10) 
        z_phase = torch.cat(GatherLayer.apply(z_phase), dim=0)  # (2N) x dim
        assert z_phase.size(0) % 2 == 0

        N = z_amp.size(0) // 2
        s_amp = torch.matmul(z_amp, z_amp.permute(1, 0))  # (2N)x(2N)
        s_phase = torch.matmul(z_phase, z_phase.permute(1, 0))
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1) 
        s_amp = torch.masked_select(s_amp, mask == 1).reshape(s_amp.size(0), -1) 
        positive_amp = s_amp[pos_ind].unsqueeze(1)  # (2N)x1
        s_phase = torch.masked_select(s_phase, mask == 1).reshape(s_phase.size(0), -1) 
        positive_phase = s_phase[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative_amp = torch.masked_select(s_amp, neg_mask == 1).reshape(s_amp.size(0), -1)
        negative_phase = torch.masked_select(s_phase, neg_mask == 1).reshape(s_phase.size(0), -1)
        losses = self.head(positive_amp,positive_phase,negative_amp,negative_phase)
        return losses



