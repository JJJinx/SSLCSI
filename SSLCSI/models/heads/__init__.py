# Copyright (c) OpenMMLab. All rights reserved.
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead
# csi heads
from .cls_csi_head import ClsCsiHead,MultiCSIClsHead,ClsCsiHead_Dual,MultiCSIClsHead_Dual
from .mae_head_csi import (MAEPretrainHead_CSI,MAEFinetuneHead_CSI,MAELinprobeHead_CSI,MAEMultilayerHead_CSI,
                            MAEPretrainHead_CSI_ARC,MAEPretrainHead_CSI_Dual,MAEPretrainHead_CSI_ARC_Dual,
                            MAELinprobeHead_CSI_Dual,MAEMultilayerHead_CSI_Dual)
from .mocov3_head_csi import MoCoV3Head_CSI

__all__ = [
    'ContrastiveHead','ClsCsiHead_Dual', 'LatentPredictHead', 'LatentClsHead','MultiCSIClsHead_Dual', 
    'MoCoV3Head', 'SimMIMHead', 'SwAVHead',
    'ClsCsiHead','MultiCSIClsHead','ClsCsiHead_Dual','MultiCSIClsHead_Dual',
    'MAEPretrainHead_CSI','MAEFinetuneHead_CSI','MAELinprobeHead_CSI','MAEMultilayerHead_CSI',
    'MAEPretrainHead_CSI_ARC','MAEPretrainHead_CSI_Dual','MAEPretrainHead_CSI_ARC_Dual',
    'MAELinprobeHead_CSI_Dual','MAEMultilayerHead_CSI_Dual','MoCoV3Head_CSI',
]
