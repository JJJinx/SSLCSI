import torch
import torch.nn as nn
import numpy as np

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCo_CSI(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCo_CSI, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input images of shape (N, C, H, W).
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
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        im_q = csi[0]
        im_k = csi[1]
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)

        # update the queue
        self._dequeue_and_enqueue(k)

        return losses


@ALGORITHMS.register_module()
class MoCo_CSI_ARC(BaseModel):
    """MoCo for CSI data with ARC.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCo_CSI_ARC, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input images of shape (N, C, H, W).
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
                (N, A, C, T) or (N,C,T). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        rand_ant1,rand_ant2 = np.random.randint(low=0, high=csi[0].size(dim=1), size=2)
        csi_q = csi[0][:,rand_ant1,:,:].unsqueeze(1).contiguous()  # q  and k are two view indeed
        csi_k = csi[1][:,rand_ant2,:,:].unsqueeze(1).contiguous() #shape (N, A(1), C, T)
        # compute query features
        q = self.encoder_q(csi_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            csi_k, idx_unshuffle = batch_shuffle_ddp(csi_k)

            k = self.encoder_k(csi_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)

        # update the queue
        self._dequeue_and_enqueue(k)

        return losses


@ALGORITHMS.register_module()
class MoCo_CSI_Dual(BaseModel):
    """MoCo for CSI data using both amp and phase.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCo_CSI_Dual, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('amp_queue', torch.randn(feat_dim, queue_len))
        self.register_buffer('phase_queue', torch.randn(feat_dim, queue_len))
        self.amp_queue = nn.functional.normalize(self.amp_queue, dim=0)
        self.phase_queue = nn.functional.normalize(self.phase_queue, dim=0)
        self.register_buffer('amp_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('phase_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, amp_keys,phase_keys):
        """Update queue."""
        # gather keys before updating queue
        amp_keys = concat_all_gather(amp_keys)
        phase_keys = concat_all_gather(phase_keys)

        batch_size = amp_keys.shape[0]

        amp_ptr = int(self.amp_queue_ptr)
        phase_ptr = int(self.phase_queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.amp_queue[:, amp_ptr:amp_ptr + batch_size] = amp_keys.transpose(0, 1)
        amp_ptr = (amp_ptr + batch_size) % self.queue_len  # move pointer
        self.phase_queue[:, phase_ptr:phase_ptr + batch_size] = phase_keys.transpose(0, 1)
        phase_ptr = (phase_ptr + batch_size) % self.queue_len  # move pointer

        self.amp_queue_ptr[0] = amp_ptr
        self.phase_queue_ptr[0] = phase_ptr

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input images of shape (N, A,C,T,2).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        assert isinstance(csi, list)
        amp_outs,phase_outs = self.backbone(csi)
        return amp_outs,phase_outs

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input images with shape
                (N, A, C, T,2). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        csi_q = csi[0]
        csi_k = csi[1]
        # compute query features
        q_amp,q_phase = self.encoder_q[0](csi_q)  
        q_amp,q_phase = self.encoder_q[1](q_amp,q_phase)  # queries: NxC

        q_amp = nn.functional.normalize(q_amp, dim=1)
        q_phase = nn.functional.normalize(q_phase, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            csi_k, idx_unshuffle = batch_shuffle_ddp(csi_k)

            k_amp,k_phase = self.encoder_k[0](csi_k)  # keys: NxC
            k_amp,k_phase = self.encoder_k[1](k_amp,k_phase)  # keys: NxC

            k_amp = nn.functional.normalize(k_amp, dim=1)
            k_phase = nn.functional.normalize(k_phase, dim=1)

            # undo shuffle
            k_amp = batch_unshuffle_ddp(k_amp, idx_unshuffle)
            k_phase = batch_unshuffle_ddp(k_phase, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_amp = torch.einsum('nc,nc->n', [q_amp, k_amp]).unsqueeze(-1)
        l_pos_phase = torch.einsum('nc,nc->n', [q_phase, k_phase]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_amp = torch.einsum('nc,ck->nk', [q_amp, self.amp_queue.clone().detach()])
        l_neg_phase = torch.einsum('nc,ck->nk', [q_phase, self.phase_queue.clone().detach()])

        losses_amp = self.head(l_pos_amp, l_neg_amp)['loss']
        losses_phase = self.head(l_pos_phase, l_neg_phase)['loss']
        # update the queue
        self._dequeue_and_enqueue(k_amp,k_phase)
        losses = dict()
        losses['loss'] = losses_amp+losses_phase
        return losses


@ALGORITHMS.register_module()
class MoCo_CSI_ARC_Dual(BaseModel):
    """MoCo with ARC using both amp and phase.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCo_CSI_ARC_Dual, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('amp_queue', torch.randn(feat_dim, queue_len))
        self.register_buffer('phase_queue', torch.randn(feat_dim, queue_len))
        self.amp_queue = nn.functional.normalize(self.amp_queue, dim=0)
        self.phase_queue = nn.functional.normalize(self.phase_queue, dim=0)
        self.register_buffer('amp_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('phase_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, amp_keys,phase_keys):
        """Update queue."""
        # gather keys before updating queue
        amp_keys = concat_all_gather(amp_keys)
        phase_keys = concat_all_gather(phase_keys)

        batch_size = amp_keys.shape[0]

        amp_ptr = int(self.amp_queue_ptr)
        phase_ptr = int(self.phase_queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.amp_queue[:, amp_ptr:amp_ptr + batch_size] = amp_keys.transpose(0, 1)
        amp_ptr = (amp_ptr + batch_size) % self.queue_len  # move pointer
        self.phase_queue[:, phase_ptr:phase_ptr + batch_size] = phase_keys.transpose(0, 1)
        phase_ptr = (phase_ptr + batch_size) % self.queue_len  # move pointer

        self.amp_queue_ptr[0] = amp_ptr
        self.phase_queue_ptr[0] = phase_ptr

    def extract_feat(self, csi):
        """Function to extract features from backbone.

        Args:
            csi (Tensor): Input images of shape (N, A,C,T,2).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensors]: backbone outputs.
        """
        amp_outs,phase_outs = self.backbone(csi)
        return amp_outs,phase_outs

    def forward_train(self, csi, **kwargs):
        """Forward computation during training.

        Args:
            csi (list[Tensor]): A list of input images with shape
                (N, A, C, T,2). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(csi, list)
        csi_q = csi[0][:,0,:,:,:].unsqueeze(1).contiguous()  # q  and k are two view indeed
        csi_k = csi[1][:,1,:,:,:].unsqueeze(1).contiguous()
        # compute query features
        q_amp,q_phase = self.encoder_q[0](csi_q)  
        q_amp,q_phase = self.encoder_q[1](q_amp,q_phase)  # queries: NxC

        q_amp = nn.functional.normalize(q_amp, dim=1)
        q_phase = nn.functional.normalize(q_phase, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            csi_k, idx_unshuffle = batch_shuffle_ddp(csi_k)

            k_amp,k_phase = self.encoder_k[0](csi_k)  # keys: NxC
            k_amp,k_phase = self.encoder_k[1](k_amp,k_phase)  # keys: NxC

            k_amp = nn.functional.normalize(k_amp, dim=1)
            k_phase = nn.functional.normalize(k_phase, dim=1)

            # undo shuffle
            k_amp = batch_unshuffle_ddp(k_amp, idx_unshuffle)
            k_phase = batch_unshuffle_ddp(k_phase, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_amp, k_amp]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_amp, self.amp_queue.clone().detach()])
        losses_amp = self.head(l_pos, l_neg)['loss']

        l_pos = torch.einsum('nc,nc->n', [q_phase, k_phase]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_phase, self.phase_queue.clone().detach()])
        losses_phase = self.head(l_pos, l_neg)['loss']
        # update the queue
        self._dequeue_and_enqueue(k_amp,k_phase)
        losses = dict()
        losses['loss'] = losses_amp+losses_phase
        return losses