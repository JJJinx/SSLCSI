import torch
import numpy as np

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class SwAV_CSI(BaseModel):
    """SwAV for CSI data.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.
    The queue is built in `core/hooks/swav_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 multi_res=True,
                 init_cfg=None,
                 **kwargs):
        super(SwAV_CSI, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        self.multi_res = multi_res
    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input csi record of shape (N, C, T) or (N, A, C, T).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(csi)
        return x

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input images with shape
                (N, C, T) or (N, A, C, T). 

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        if self.multi_res:
            # multi-res forward passes
            idx_crops = torch.cumsum(
                torch.unique_consecutive(
                    torch.tensor([i.shape[-1] for i in csi]),
                    return_counts=True)[1], 0)
            start_idx = 0
            output = []
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(csi[start_idx:end_idx])) #after feeding into backbone it becomes [N*views,C,H,W]
                output.append(_out)
                start_idx = end_idx
        else:
            output = []
            _out = self.backbone(torch.cat(csi))
            output.append(_out)
        output = self.neck(output)[0]
        loss = self.head(output)
        return loss


@ALGORITHMS.register_module()
class SwAV_CSI_ARC(BaseModel):
    """SwAV for CSI data with ARC.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 multi_res=True,
                 init_cfg=None,
                 **kwargs):
        super(SwAV_CSI_ARC, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        self.multi_res = multi_res
    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input csi record of shape (N, A, C, T).
                Typically these should be mean centered and std scaled.
                 List length equal to the sum of num_view 8 (2,6) TODO

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        assert isinstance(csi, list)
        print(len(csi))
        raise RuntimeError
        #csi=csi[0]
        # rand_ant1,rand_ant2 = np.random.randint(low=0, high=3, size=2)
        # csi_ant1 = csi[0][:,rand_ant1,:,:]
        # csi_ant2 = csi[1][:,rand_ant2,:,:] # shape [N,1, C, T]
        # if 'ResNet' not in self.backbone.__class__.__name__:
        #     csi_ant1 = csi_ant1.squeeze()
        #     csi_ant2 = csi_ant2.squeeze() #shape [N, C, T]

        # csi = torch.stack([csi_ant1,csi_ant2], 1) #[N, 2, C, T]  or  [N, 2,1, C, T]
        
        # if 'ResNet' not in self.backbone.__class__.__name__:
        #     csi = csi.reshape(
        #     (csi.size(0) * 2, csi.size(2), csi.size(3))) # shape [num_view*N, C, T ] 
        # else:
        #     csi = csi.unsqueeze(2)  #[N, 2,1, C, T]
        #     csi = csi.reshape(
        #     (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape  [2N, 1, C, T]

        x = self.backbone(csi)
        return x

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input images with shape
                (N, A, C, T). List length equal to the sum of num_view 8 (2,6) TODO

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        #csi=csi[0]
        if self.multi_res:
            rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi[0].size(dim=1), size=2)
            if 'ResNet' not in self.backbone.__class__.__name__:
                csi[0] = csi[0][:,rand_ant1,:,:].squeeze()
                csi[1] = csi[1][:,rand_ant2,:,:].squeeze()
                for i in range(2,8):
                    csi[i] = csi[i][:,np.random.randint(low=0, high=csi[0].size(dim=1), size=1),:,:].squeeze()#shape [N, C, T]
            else:
                csi[0] = csi[0][:,rand_ant1,:,:].unsqueeze(1)
                csi[1] = csi[1][:,rand_ant2,:,:].unsqueeze(1) # [N,1,C,T]([32, 30, 500])
                for i in range(2,8):
                    csi[i] = csi[i][:,np.random.randint(low=0, high=csi[0].size(dim=1), size=1),:,:]#shape [N, 1, C, T]
            # multi-res forward passes
            idx_crops = torch.cumsum(
                torch.unique_consecutive(
                    torch.tensor([i.shape[-1] for i in csi]),
                    return_counts=True)[1], 0)
            start_idx = 0
            output = []
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(csi[start_idx:end_idx]))
                output.append(_out)
                start_idx = end_idx
        else:
            rand_ant1,rand_ant2 = np.random.randint(low=0,high=csi[0].size(dim=1), size=2)
            if 'ResNet' not in self.backbone.__class__.__name__:
                csi[0] = csi[0][:,rand_ant1,:,:].squeeze()
                csi[1] = csi[1][:,rand_ant2,:,:].squeeze()
                print(csi[0].shape)
                csi = torch.stack([csi_ant1,csi_ant2], 1) #[N, 2, C, T] 
                print(csi.shape)
                raise RuntimeError
                csi = csi.reshape(
                (csi.size(0) * 2, csi.size(2), csi.size(3))) # shape [num_view*N, C, T ] 

            else:
                csi[0] = csi[0][:,rand_ant1,:,:]
                csi[1] = csi[1][:,rand_ant2,:,:]
                print(csi[0].shape)
                csi = torch.stack([csi_ant1,csi_ant2], 1) # [N, 2,1, C, T]
                print(csi.shape)
                raise RuntimeError
                csi = csi.unsqueeze(2)  #[N, 2,1, C, T]
                csi = csi.reshape(
                (csi.size(0) * 2, csi.size(2), csi.size(3), csi.size(4))) # shape  [2N, 1, C, T]

            output = []
            _out = self.backbone(csi)
            output.append(_out)
            
        output = self.neck(output)[0]
        loss = self.head(output)
        return loss