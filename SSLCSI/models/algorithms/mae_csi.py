from typing import Dict, Optional, Tuple
import numpy as np
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MAE_CSI(BaseModel):
    """MAE for CSI data.
    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(MAE_CSI, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(MAE_CSI, self).init_weights()

    def extract_feat(self, csi):
        """Function to extract features from backbone.
        Args:
            csi (list[Tensor]): Input csi record of shape (N, A, C, T).
        Returns:
            tuple[Tensor]: backbone outputs.
        """
        csi = csi[0]
        return self.backbone(csi)

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input csi with shape
                (N, A, C, T). 
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]
        latent, mask, ids_restore = self.backbone(csi)
        pred = self.neck(latent, ids_restore) # tensor

        losses = self.head(csi, pred, mask)

        return losses


    def forward_test(self, csi, **kwargs):
        """Forward computation during testing.

        Args:
            csi (list[torch.Tensor]): Input images of shape (N, A, C, T).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        csi = csi[0]
        latent, mask, ids_restore = self.backbone(csi)
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred


@ALGORITHMS.register_module()
class MAE_CSI_ARC(BaseModel):
    """MAE with ARC for CSI data.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(MAE_CSI_ARC, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(MAE_CSI_ARC, self).init_weights()

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input csi record of shape (N, A, C, T).
             list length should be 1.
        Returns:
            tuple[Tensor]: backbone outputs.
        """

        csi = csi[0]
        rand_ant1,_ = np.random.randint(low=0, high=3, size=2)
        csi_ant1 = csi[:,rand_ant1,:,:].unsqueeze(1) # shape [N, 1, C, T] [32, 1, 30, 500]

        return self.backbone(csi_ant1)

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input csi with shape
                (N, A, C, T). list length should be 1.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi.size(1), size=2)
        #numbers = np.arange(csi.size(1))
        #np.random.shuffle(numbers)
        #rand_ant1,rand_ant2 = numbers[:2],numbers[2:]
        csi_ant1 = csi[:,rand_ant1,:,:].unsqueeze(1)
        csi_ant2 = csi[:,rand_ant2,:,:].unsqueeze(1) # shape [N, 1, C, T] [32, 1, 30, 500]
        if len(csi_ant1.shape)>4:
            csi_ant1 = csi_ant1.squeeze()
            csi_ant2 = csi_ant2.squeeze()
        latent, mask, ids_restore = self.backbone(csi_ant1) # mask shape [N,num_patches] [64, 150]
        latent2,_,_ = self.backbone(csi_ant2)

        pred = self.neck(latent, ids_restore)#shape torch.Size([64, 150, neck.inchans*patch**2])
        #losses = self.head(csi_ant2, pred, mask) 
        losses = self.head(csi_ant1, pred, mask,latent,latent2) # loss=mse(backbone(csi_ant1),csi_ant1)+alpha similarity(csi_ant1,csi_ant2)

        return losses


    def forward_test(self, csi, **kwargs):
        """Forward computation during testing.

        Args:
            csi (list[torch.Tensor]): Input images of shape (N, A, C, T).
             list length should be 1.
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        csi = csi[0]
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi.size(1), size=2)
        csi_ant1 = csi[:,rand_ant1,:,:].unsqueeze(1)
        csi_ant2 = csi[:,rand_ant2,:,:].unsqueeze(1) # shape [N, 1, C, T] [32, 1, 30, 500]

        latent, mask, ids_restore = self.backbone(csi_ant1)
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred


@ALGORITHMS.register_module()
class MAE_CSI_Dual(BaseModel):
    """MAE. Use distinct backbone and classifier to process the amp and phase input. 
    Fuse them at the final layer    

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(MAE_CSI_Dual, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(MAE_CSI_Dual, self).init_weights()

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input csi record of shape (N, A, C, T,2).
             list length should be 1.
        Returns:
            tuple[Tensor]: backbone outputs.
        """

        csi = csi[0]
        return self.backbone(csi)

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input csi with shape
                (N,A,C,T,2(amp,ratio_ang)). list length should be 1.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]
        amp_latent, amp_mask, amp_ids_restore,phase_latent,phase_mask,phase_ids_restore= \
            self.backbone(csi) 
        amp_pred = self.neck(amp_latent, amp_ids_restore)#shape torch.Size([64, 150, neck.inchans*patch**2])
        phase_pred = self.neck(phase_latent,phase_ids_restore)
        #losses = self.head(csi_ant2, pred, mask) 
        losses = self.head(csi[:,:,:,:,0],csi[:,:,:,:,1],amp_pred,phase_pred,amp_mask,phase_mask) 
        return losses


    def forward_test(self, csi, **kwargs):
        """Forward computation during testing.

        Args:
            csi (list[torch.Tensor]): Input images of shape (N, A, C, T,2).
             list length should be 1.
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        # csi = csi[0]
        # latent, mask, ids_restore = self.backbone(csi)
        # pred = self.neck(latent, ids_restore)

        # pred = self.head.unpatchify(pred)
        # pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        # mask = mask.detach()
        # mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
        #                                  3)  # (N, H*W, p*p*3)
        # mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        csi = csi[0]
        amp_latent, amp_mask, amp_ids_restore,phase_latent,phase_mask,phase_ids_restore= \
            self.backbone(csi) 

        amp_pred = self.neck(amp_latent, amp_ids_restore)#shape torch.Size([64, 150, neck.inchans*patch**2])
        phase_pred = self.neck(phase_latent,phase_ids_restore)

        amp_pred = self.head.unpatchify(amp_pred)
        amp_pred = torch.einsum('nchw->nhwc', amp_pred).detach().cpu()
        phase_pred = self.head.unpatchify(phase_pred)
        phase_pred = torch.einsum('nchw->nhwc', phase_pred).detach().cpu()

        amp_mask = amp_mask.detach()
        amp_mask = amp_mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         2)  # (N, H*W, p*p*3)
        amp_mask = self.head.unpatchify(amp_mask)  # 1 is removing, 0 is keeping
        amp_mask = torch.einsum('nchw->nhwc', amp_mask).detach().cpu()
        #########
        phase_mask = phase_mask.detach()
        phase_mask = phase_mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         2)  # (N, H*W, p*p*3) why is 3 here if it is the ant it should be 2
        phase_mask = self.head.unpatchify(phase_mask)  # 1 is removing, 0 is keeping
        phase_mask = torch.einsum('nchw->nhwc', phase_mask).detach().cpu()

        return amp_mask,phase_mask, amp_pred,phase_pred


@ALGORITHMS.register_module()
class MAE_CSI_ARC_Dual(BaseModel):
    """MAE with ARC using both amp and amplitude.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(MAE_CSI_ARC_Dual, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(MAE_CSI_ARC_Dual, self).init_weights()

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (list[Tensor]): Input csi record of shape (N, A, C, T,2).
             list length should be 1.
        Returns:
            tuple[Tensor]: backbone outputs.
        """

        csi = csi[0]
        rand_ant1,_ = np.random.randint(low=0, high=2, size=2)
        csi_ant1 = csi[:,rand_ant1,:,:,:].unsqueeze(1) # shape [N, 1, C, T,2] [32, 1, 30, 500]

        return self.backbone(csi_ant1)

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input csi with shape
                (N,A,C,T,2(amp,amp,ratio_ang)). list length should be 1.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        csi = csi[0]
        csi_ant1 = csi[:,0,:,:,:].unsqueeze(1)
        csi_ant2 = csi[:,1,:,:,:].unsqueeze(1) # shape [N, 1, C, T,2]  [64, 1, 30, 500, 2]
        amp_latent1, amp_mask, amp_ids_restore,phase_latent1,phase_mask,phase_ids_restore= \
            self.backbone(csi_ant1) 
        amp_latent2, _, _,phase_latent2,_,_ = self.backbone(csi_ant2)

        amp_pred = self.neck(amp_latent1, amp_ids_restore)#shape torch.Size([64, 150, neck.inchans*patch**2])
        phase_pred = self.neck(phase_latent1,phase_ids_restore)

        #losses = self.head(csi_ant2, pred, mask) 
        losses = self.head(csi_ant1[:,:,:,:,0],csi_ant1[:,:,:,:,1], amp_pred,phase_pred,amp_mask,phase_mask,
                amp_latent1,amp_latent2,phase_latent1,phase_latent2) # loss=mse(backbone(csi_ant1),csi_ant1)+alpha similarity(csi_ant1,csi_ant2)

        return losses
 

    def forward_test(self, csi, **kwargs):
        """Forward computation during testing.

        Args:
            csi (list[torch.Tensor]): Input images of shape (N, A, C, T,2).
             list length should be 1.
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        csi = csi[0]
        csi_ant1 = csi[:,0,:,:,:].unsqueeze(1)
        csi_ant2 = csi[:,1,:,:,:].unsqueeze(1) # shape [N, 1, C, T,2] 
        amp_latent1, amp_mask, amp_ids_restore,phase_latent1,phase_mask,phase_ids_restore= \
            self.backbone(csi_ant1) 
        amp_latent2, _, _,phase_latent2,_,_ = self.backbone(csi_ant2)

        amp_pred = self.neck(amp_latent1, amp_ids_restore)#shape torch.Size([64, 150, neck.inchans*patch**2])
        phase_pred = self.neck(phase_latent1,phase_ids_restore)

        amp_pred = self.head.unpatchify(amp_pred)
        amp_pred = torch.einsum('nchw->nhwc', amp_pred).detach().cpu()
        phase_pred = self.head.unpatchify(phase_pred)
        phase_pred = torch.einsum('nchw->nhwc', phase_pred).detach().cpu()

        amp_mask = amp_mask.detach()
        amp_mask = amp_mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         2)  # (N, H*W, p*p*3)
        amp_mask = self.head.unpatchify(amp_mask)  # 1 is removing, 0 is keeping
        amp_mask = torch.einsum('nchw->nhwc', amp_mask).detach().cpu()
        #########
        phase_mask = phase_mask.detach()
        phase_mask = phase_mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         2)  # (N, H*W, p*p*3)
        phase_mask = self.head.unpatchify(phase_mask)  # 1 is removing, 0 is keeping
        phase_mask = torch.einsum('nchw->nhwc', phase_mask).detach().cpu()

        return amp_mask,phase_mask, amp_pred,phase_pred