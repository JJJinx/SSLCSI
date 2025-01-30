# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseModel
from .classification import Classification
from .mmcls_classifier_wrapper import MMClsImageClassifierWrapper
from .mae_csi import MAE_CSI,MAE_CSI_ARC,MAE_CSI_ARC_Dual,MAE_CSI_Dual
from .classification_csi_resnet import Classification_CSI_ResNet,Classification_CSI_ResNet_ARC,Classification_CSI_ResNet_Dual,Classification_CSI_ResNet_ARC_Dual
from .classification_csi_vit import Classification_CSI_VIT,Classification_CSI_VIT_ARC,Classification_CSI_VIT_ARC_Dual
from .swav_csi import SwAV_CSI,SwAV_CSI_ARC
from .relative_loc_csi import RelativeLoc_CSI,RelativeLoc_CSI_ARC
from .simclr_csi import SimCLR_CSI,SimCLR_CSI_ARC,SimCLR_CSI_Dual,SimCLR_CSI_ARC_Dual
from .moco_csi import MoCo_CSI,MoCo_CSI_ARC,MoCo_CSI_Dual,MoCo_CSI_ARC_Dual
from .mocov3_csi import MoCoV3_CSI

__all__ = [
    'BaseModel', 'Classification',
    'MAE', 'MoCoV3', 'SimMIM', 'MMClsImageClassifierWrapper',
    'SimCLR_CSI', 'SimCLR_CSI_ARC','SimCLR_CSI_Dual','SimCLR_CSI_ARC','SimCLR_CSI_ARC_Dual',
    'MAE_CSI','MAE_CSI_ARC','MAE_CSI_Dual','MAE_CSI_ARC_Dual', 'SwAV_CSI','SwAV_CSI_ARC',
    'RelativeLoc_CSI','RelativeLoc_CSI_ARC','MoCo_CSI','MoCo_CSI_ARC','MoCo_CSI_Dual','MoCo_CSI_ARC_Dual','MoCoV3_CSI',
    'Classification_CSI_ResNet','Classification_CSI_ResNet_ARC','Classification_CSI_ResNet_Dual','Classification_CSI_ResNet_ARC_Dual',
    'Classification_CSI_VIT','Classification_CSI_VIT_ARC','Classification_CSI_VIT_ARC_Dual',
    'GKDE_ResNet','GKDE_ResNet_ARC',
]
