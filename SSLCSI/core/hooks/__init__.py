# Copyright (c) OpenMMLab. All rights reserved.
from .cosineAnnealing_hook import StepFixCosineAnnealingLrUpdaterHook
from .deepcluster_hook import DeepClusterHook
from .densecl_hook import DenseCLHook
from .momentum_update_hook import MomentumUpdateHook
from .odc_hook import ODCHook
from .optimizer_hook import DistOptimizerHook, GradAccumFp16OptimizerHook
from .simsiam_hook import SimSiamHook
from .swav_hook import SwAVHook
from .swa_hook import SWAHook
from .SetEpochInfo_hook import SetEpochInfoHook
from .customized_TensorboardLoggerHook import CustomTensorboardLoggerHook

__all__ = [
    'MomentumUpdateHook', 'DeepClusterHook', 'DenseCLHook', 'ODCHook',
    'DistOptimizerHook', 'GradAccumFp16OptimizerHook', 'SimSiamHook',
    'SwAVHook', 'StepFixCosineAnnealingLrUpdaterHook',
    'SWAHook', 'SetEpochInfoHook','CustomTensorboardLoggerHook'
]
