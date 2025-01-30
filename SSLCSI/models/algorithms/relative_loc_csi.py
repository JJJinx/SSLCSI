import torch
import numpy as np

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class RelativeLoc_CSI(BaseModel):
    """Relative patch location for CSI data. 
    We regard CSI data which has a shape of [A,C,T] as a image with height C width T and A channel. 
    Then, we randomly select 9 non-overlapping patches and feed them to the algorithms.

    Implementation of `Unsupervised Visual Representation Learning
    by Context Prediction <https://arxiv.org/abs/1505.05192>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(RelativeLoc_CSI, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input CSI signal of shape (8*N, 2*A, C, T).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(csi)
        return x

    def forward_train(self, csi, patch_label, **kwargs):
        """Forward computation during training.

        Args:
            csi (Tensor): Input CSI signal of shape (8*N, 2*A, C, T). it contains both query patch and anchor patch.
            patch_label (Tensor): Labels for the relative patch locations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #The chunk function divides the tensor into n equal parts along a certain dimension. 
        #Here it is divided into two parts, namely query and anchor.
        csi1, csi2 = torch.chunk(csi, 2, dim=1) 
        if self.backbone.__class__.__name__=='CausaulNet':
            N,ATTENA,C,T =csi1.size() 
            csi1 = csi1.reshape(N,-1,T)
            csi2 = csi2.reshape(N,-1,T)
        x1 = self.extract_feat(csi1)  # tuple shape [8N,512,1,4]
        x2 = self.extract_feat(csi2)  # tuple
        # After concatenation, perform 8-class classification.
        x = (torch.cat((x1[0], x2[0]), dim=1), ) #x shape[1024, 1024, 1, 4]
        x = self.neck(x)
        outs = self.head(x)
        loss_inputs = (outs, patch_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (Tensor): Input CSI of shape (8*N, 2*A, C, T).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi1, csi2 = torch.chunk(csi, 2, dim=1)
        if self.backbone.__class__.__name__=='CausaulNet':
            N,ATTENA,C,T =csi1.size() 
            csi1 = csi1.reshape(N,-1,T)
            csi2 = csi2.reshape(N,-1,T)

        x1 = self.extract_feat(csi1)  # tuple
        x2 = self.extract_feat(csi2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]
        return dict(zip(keys, out_tensors))

    def forward(self, img, patch_label=None, mode='train', **kwargs):
        """Forward function to select mode and modify the input image shape.

        Args:
            img (Tensor): Input csi signal, the shape depends on mode.
                Typically these should be mean centered and std scaled.
                Using 'img' to represent csi signal to keep consistent with other algorithm.
        """
        csi = img  #[128, 8, 6, 13, 123] There are 8 pairs of patches in each batch, each pair with one anchor.
        if mode != 'extract' and csi.dim() == 5:  # Nx8x(2C)xHxW
            assert patch_label.dim() == 2  # Nx8
            csi = csi.view(
                csi.size(0) * csi.size(1), csi.size(2), csi.size(3),
                csi.size(4))  # (8N)x(2C)xHxW
            patch_label = torch.flatten(patch_label)  # (8N)
        if mode == 'train':
            return self.forward_train(csi, patch_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(csi, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(csi)
        else:
            raise Exception(f'No such mode: {mode}')



@ALGORITHMS.register_module()
class RelativeLoc_CSI_ARC(BaseModel):
    """Relative patch location with ARC.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(RelativeLoc_CSI_ARC, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input CSI signal of shape (8*N, 2*A, C, T).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(csi)
        return x

    def forward_train(self, csi, patch_label, **kwargs):
        """Forward computation during training.

        Args:
            csi (Tensor): Input CSI signal of shape (8*N, 2*A, C, T). it contains both query patch and anchor patch.
            patch_label (Tensor): Labels for the relative patch locations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #csi shape [8*N,2A,C,T]  [8N, 2A, 29, 470] csida
        csi1, csi2 = torch.chunk(csi, 2, dim=1) 
        # csi1 [8N,A,C,T]
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi1.size(dim=1), size=2)
        csi1 = csi1[:,rand_ant1,:,:].unsqueeze(1).contiguous()
        csi2 = csi2[:,rand_ant2,:,:].unsqueeze(1).contiguous()# TODO
        if self.backbone.__class__.__name__=='CausaulNet':
            N,ATTENA,C,T =csi1.size() 
            csi1 = csi1.reshape(N,-1,T)
            csi2 = csi2.reshape(N,-1,T)
        x1 = self.extract_feat(csi1)  # tuple shape [8N,512,1,4]
        x2 = self.extract_feat(csi2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1), ) #x shape[1024, 1024, 1, 4]
        x = self.neck(x)
        outs = self.head(x)
        loss_inputs = (outs, patch_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (Tensor): Input images of shape (8*N, 2*A, C, T).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        csi1, csi2 = torch.chunk(csi, 2, dim=1)
        if self.backbone.__class__.__name__=='CausaulNet':
            N,ATTENA,C,T =csi1.size() 
            csi1 = csi1.reshape(N,-1,T)
            csi2 = csi2.reshape(N,-1,T)
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi1.size(dim=1), size=2)
        csi1 = csi1[:,rand_ant1,:,:].unsqueeze(1).contiguous()
        csi2 = csi2[:,rand_ant2,:,:].unsqueeze(1).contiguous()# TODO
        x1 = self.extract_feat(csi1)  # tuple
        x2 = self.extract_feat(csi2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]
        return dict(zip(keys, out_tensors))

    def forward(self, img, patch_label=None, mode='train', **kwargs):
        """Forward function to select mode and modify the input image shape.

        Args:
            img (Tensor): Input csi signal, the shape depends on mode.
                Typically these should be mean centered and std scaled.
                Using 'img' to represent csi signal to keep consistent with other algorithm.
        """
        csi = img  #[128, 8, 6, 13, 123] 
        if mode != 'extract' and csi.dim() == 5:  # Nx8x(2C)xHxW
            assert patch_label.dim() == 2  # Nx8
            csi = csi.view(
                csi.size(0) * csi.size(1), csi.size(2), csi.size(3),
                csi.size(4))  # (8N)x(2C)xHxW
            patch_label = torch.flatten(patch_label)  # (8N)
        if mode == 'train':
            return self.forward_train(csi, patch_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(csi, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(csi)
        else:
            raise Exception(f'No such mode: {mode}')