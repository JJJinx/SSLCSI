import torch
from mmcls.models import LabelSmoothLoss
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from torch import nn
from functools import partial

from ..builder import HEADS
from ..utils import accuracy
from ..utils.csi_utils.loss_for_long_tail import FocalLoss,balanced_softmax_loss


@HEADS.register_module()
class MAEPretrainHead_CSI(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16):
        super(MAEPretrainHead_CSI, self).__init__()
        self.norm_pix = norm_pix
        if isinstance(patch_size,int):
            self.patch_size=(patch_size,patch_size)
        else:
            self.patch_size = patch_size

    def patchify(self, csi):
        p = self.patch_size
        # assert csi.shape[2] == csi.shape[3] and csi.shape[2] % p == 0
        # csi shape [batch,A,30,T]
        assert csi.shape[2] % p[0] == 0 
        assert csi.shape[3] % p[1] == 0 
        h = csi.shape[2] // p[0]
        w = csi.shape[3] // p[1]
        x = csi.reshape(shape=(csi.shape[0], csi.shape[1], h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(csi.shape[0], h * w, p[0]*p[1] * csi.shape[1]))
        return x

    def forward(self, x, pred, mask):
        """
            input x shape  [N ,A,C,T]
            input pred shape  [N,num_patches,patch_size**2*A]
        """
        losses = dict()
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss
        return losses


@HEADS.register_module()
class MAEFinetuneHead_CSI(BaseModule):
    """Fine-tuning head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
        norm_layer: (nn.Module): normalization layer.
    """

    def __init__(self, 
                embed_dim, 
                num_classes=1000, 
                label_smooth_val=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                global_pool=False):
        super(MAEFinetuneHead_CSI, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.criterion = LabelSmoothLoss(label_smooth_val, num_classes)
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)
        self.global_pool=global_pool

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        outputs = self.head(outcome)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses


@HEADS.register_module()
class MAELinprobeHead_CSI(BaseModule):
    """Linear probing head for MAE.
    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000,focal_loss_flag=False):
        super(MAELinprobeHead_CSI,self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss_flag = focal_loss_flag
        self.criterion_focal = FocalLoss()

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)

    def forward(self, x):
        """"Get the logits."""
        #x = self.bn(x)
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        if self.focal_loss_flag:
            losses['loss'] = self.criterion_focal(outputs[0], labels)
            #losses['loss'] = balanced_softmax_loss(labels,outputs[0])
        else:
            losses['loss'] = self.criterion(outputs[0], labels)

        return losses


@HEADS.register_module()
class MAEMultilayerHead_CSI(BaseModule):
    """Multi-layer head for MAE.
    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000,focal_loss_flag=False):
        super(MAEMultilayerHead_CSI,self).__init__()
        self.fc0 = nn.Linear(embed_dim, embed_dim//2)
        self.head = nn.Linear(embed_dim//2, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim//2, affine=False, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss_flag = focal_loss_flag
        self.criterion_focal = FocalLoss()
        self.num_classes = num_classes

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.fc0.bias, 0)
        trunc_normal_(self.fc0.weight, std=0.01)

    def forward(self, x):
        """"Get the logits."""
        x = self.fc0(x)
        x = self.bn(x)
        x = self.relu(x)
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        if self.focal_loss_flag:
            losses['loss'] = self.criterion_focal(outputs[0], labels)
            #losses['loss'] = balanced_softmax_loss(labels,outputs[0])
        else:
            losses['loss'] = self.criterion(outputs[0], labels)

        return losses


@HEADS.register_module()
class MAEPretrainHead_CSI_ARC(BaseModule):
    """Pre-training head for MAE_ARC.
        loss = mse+similarity
    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16,alpha=0.5):
        super(MAEPretrainHead_CSI_ARC, self).__init__()
        self.norm_pix = norm_pix
        if isinstance(patch_size,int):
            self.patch_size=(patch_size,patch_size)
        else:
            self.patch_size = patch_size
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = alpha

    def patchify(self, csi):
        p = self.patch_size
        # assert csi.shape[2] == csi.shape[3] and csi.shape[2] % p == 0
        # csi shape [batch,A,30,T]
        assert csi.shape[2] % p[0] == 0 
        assert csi.shape[3] % p[1] == 0 
        h = csi.shape[2] // p[0]
        w = csi.shape[3] // p[1]
        x = csi.reshape(shape=(csi.shape[0], csi.shape[1], h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(csi.shape[0], h * w, p[0]*p[1] * csi.shape[1]))
        return x

    def forward(self, x, pred, mask,latent1,latent2):
        """
            input x shape  [N ,A,C,T]
            input pred shape  [N,num_patches,patch_size**2*A]
            input latent1, latent2 shape [N, 38, embede_dim]
        """
        #ant = 2 时 [64, 46, 768]
        losses = dict()
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1) # [64, 150]
        latent1=latent1.view(latent1.shape[0],-1)#self.avg_pool2d(latent1)
        latent2=latent2.view(latent2.shape[0],-1)#self.avg_pool2d(latent2)
        loss_simlarity = nn.functional.cosine_similarity(latent1,latent2,dim=1).sum()/latent1.shape[0]

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss-self.alpha*loss_simlarity
        return losses

@HEADS.register_module()
class MAEPretrainHead_CSI_Dual(BaseModule):
    """Pre-training head for MAE_CSI_Dual.
        loss = mse+similarity
    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16,alpha=0.5):
        super(MAEPretrainHead_CSI_Dual, self).__init__()
        self.norm_pix = norm_pix
        if isinstance(patch_size,int):
            self.patch_size=(patch_size,patch_size)
        else:
            self.patch_size = patch_size
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = alpha

    def patchify(self, csi):
        p = self.patch_size
        # assert csi.shape[2] == csi.shape[3] and csi.shape[2] % p == 0
        # csi shape [batch,A,30,T]
        assert csi.shape[2] % p[0] == 0 
        assert csi.shape[3] % p[1] == 0 
        h = csi.shape[2] // p[0]
        w = csi.shape[3] // p[1]
        x = csi.reshape(shape=(csi.shape[0], csi.shape[1], h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(csi.shape[0], h * w, p[0]*p[1] * csi.shape[1]))
        return x


    def forward(self, amp,phase, amp_pred,phase_pred,amp_mask,phase_mask):
        """
            input amp and phase shape  [N ,A,C,T]
            input pred shape  [N,num_patches,patch_size**2*A]
            input latent1, latent2 shape [N, 38, embede_dim]
        """
        losses = dict()
        ## amp
        target_amp = self.patchify(amp)
        if self.norm_pix:
            mean_amp = target_amp.mean(dim=-1, keepdim=True)
            var_amp = target_amp.var(dim=-1, keepdim=True)
            target_amp = (target_amp - mean_amp) / (var_amp + 1.e-6)**.5
        loss_amp = (amp_pred - target_amp)**2
        loss_amp = loss_amp.mean(dim=-1) # [64, 150]
        loss_amp = (loss_amp * amp_mask).sum() / amp_mask.sum()

        #phase
        target_phase = self.patchify(phase)
        if self.norm_pix:
            mean_phase = target_phase.mean(dim=-1, keepdim=True)
            var_phase = target_phase.var(dim=-1, keepdim=True)
            target_phase = (target_phase - mean_phase) / (var_phase + 1.e-6)**.5
        loss_phase = (phase_pred - target_phase)**2
        loss_phase = loss_phase.mean(dim=-1) # [64, 150]
        loss_phase = (loss_phase * phase_mask).sum() / phase_mask.sum()

        losses['loss'] = loss_amp+loss_phase
        return losses


@HEADS.register_module()
class MAEPretrainHead_CSI_ARC_Dual(BaseModule):
    """Pre-training head for MAE_ARC_CSI_Dual.
        loss = mse+similarity
    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix=False, patch_size=16,alpha=0.5):
        super(MAEPretrainHead_CSI_ARC_Dual, self).__init__()
        self.norm_pix = norm_pix
        if isinstance(patch_size,int):
            self.patch_size=(patch_size,patch_size)
        else:
            self.patch_size = patch_size
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = alpha

    def patchify(self, csi):
        p = self.patch_size
        # assert csi.shape[2] == csi.shape[3] and csi.shape[2] % p == 0
        # csi shape [batch,A,30,T]
        assert csi.shape[2] % p[0] == 0 
        assert csi.shape[3] % p[1] == 0 
        h = csi.shape[2] // p[0]
        w = csi.shape[3] // p[1]
        x = csi.reshape(shape=(csi.shape[0], csi.shape[1], h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(csi.shape[0], h * w, p[0]*p[1] * csi.shape[1]))
        return x

    def forward(self, amp,phase, amp_pred,phase_pred,amp_mask,phase_mask,
                amp_latent1,amp_latent2,phase_latent1,phase_latent2):
        """
            input amp and phase shape  [N ,A,C,T,2]
            input pred shape  [N,num_patches,patch_size**2*A]
            input latent1, latent2 shape [N, 38, embede_dim]
        """
        losses = dict()
        ## amp
        target_amp = self.patchify(amp)
        if self.norm_pix:
            mean_amp = target_amp.mean(dim=-1, keepdim=True)
            var_amp = target_amp.var(dim=-1, keepdim=True)
            target_amp = (target_amp - mean_amp) / (var_amp + 1.e-6)**.5
        loss_amp = (amp_pred - target_amp)**2
        loss_amp = loss_amp.mean(dim=-1) # [64, 150]
        amp_latent1=amp_latent1.view(amp_latent1.shape[0],-1)#self.avg_pool2d(amp_latent1)
        amp_latent2=amp_latent2.view(amp_latent2.shape[0],-1)#self.avg_pool2d(amp_latent2)
        loss_simlarity_amp = nn.functional.cosine_similarity(amp_latent1,amp_latent2,dim=1).sum()/amp_latent1.shape[0]
        loss_amp = (loss_amp * amp_mask).sum() / amp_mask.sum()

        #phase
        target_phase = self.patchify(phase)
        if self.norm_pix:
            mean_phase = target_phase.mean(dim=-1, keepdim=True)
            var_phase = target_phase.var(dim=-1, keepdim=True)
            target_phase = (target_phase - mean_phase) / (var_phase + 1.e-6)**.5
        loss_phase = (phase_pred - target_phase)**2
        loss_phase = loss_phase.mean(dim=-1) # [64, 150]
        phase_latent1=phase_latent1.view(phase_latent1.shape[0],-1)#self.avg_pool2d(phase_latent1)
        phase_latent2=phase_latent2.view(phase_latent2.shape[0],-1)#self.avg_pool2d(phase_latent2)
        loss_simlarity_phase = nn.functional.cosine_similarity(phase_latent1,phase_latent2,dim=1).sum()/phase_latent1.shape[0]
        loss_phase = (loss_phase * phase_mask).sum() / phase_mask.sum()

        losses['loss'] = loss_amp-self.alpha*loss_simlarity_amp+loss_phase-self.alpha*loss_simlarity_phase
        return losses


@HEADS.register_module()
class MAELinprobeHead_CSI_Dual(BaseModule):
    """Linear probing head for MAE.
    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super(MAELinprobeHead_CSI_Dual,self).__init__()
        self.amp_head = nn.Linear(embed_dim, num_classes)
        self.amp_bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.phase_head = nn.Linear(embed_dim, num_classes)
        self.phase_bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.Softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

    def init_weights(self):
        nn.init.constant_(self.amp_head.bias, 0)
        trunc_normal_(self.amp_head.weight, std=0.01)
        nn.init.constant_(self.phase_head.bias, 0)
        trunc_normal_(self.phase_head.weight, std=0.01)

    def forward(self, amp,phase):
        """"Get the logits."""
        # input x shape [B,dim]
        #amp = self.amp_bn(amp)
        amp_outputs = self.amp_head(amp) # [B,num_cls]
        amp_logit = self.Softmax(amp_outputs)
        #phase = self.phase_bn(phase)
        phase_outputs = self.phase_head(phase) # [B,num_cls]
        phase_logit = self.Softmax(phase_outputs)
        outputs = torch.exp(0.5*amp_logit+0.5*phase_logit)
        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(torch.log(outputs[0]), labels)

        return losses

@HEADS.register_module()
class MAEMultilayerHead_CSI_Dual(BaseModule):
    """Multi-layer head for MAE.
    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super(MAEMultilayerHead_CSI_Dual,self).__init__()
        self.amp_fc0 = nn.Linear(embed_dim, embed_dim//2)
        self.amp_head = nn.Linear(embed_dim//2, num_classes)
        self.amp_bn = nn.BatchNorm1d(embed_dim//2, affine=False, eps=1e-6)
        self.phase_fc0 = nn.Linear(embed_dim, embed_dim//2)
        self.phase_head = nn.Linear(embed_dim//2, num_classes)
        self.phase_bn = nn.BatchNorm1d(embed_dim//2, affine=False, eps=1e-6)

        self.relu = nn.ReLU(inplace=True)
        self.Softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

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
        """"Get the logits."""
        amp = self.amp_fc0(amp)
        amp = self.amp_bn(amp)
        amp = self.relu(amp)
        amp_outputs = self.amp_head(amp)
        amp_logit = self.Softmax(amp_outputs)

        phase = self.phase_fc0(phase)
        phase = self.phase_bn(phase)
        phase = self.relu(phase)
        phase_outputs = self.phase_head(phase)
        phase_logit = self.Softmax(phase_outputs)
        outputs = torch.exp(0.5*amp_logit+0.5*phase_logit)
        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(torch.log(outputs[0]), labels)

        return losses






