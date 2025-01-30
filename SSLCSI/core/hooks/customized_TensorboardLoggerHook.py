# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmcv.parallel import is_module_wrapper
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class CustomTensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)
    
    def after_train_iter(self, runner) -> None:
        """Log intermediate variables directly from the model's forward pass to TensorBoard"""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
            self.l_pos_length = len(model.head.l_pos)
            self.l_neg_length = len(model.head.l_neg)
        if hasattr(model.head, 'del_neg_mask'):
            self.del_neg_mask = model.head.del_neg_mask
        if hasattr(model.head, 'neg_to_pos_mask'):
            self.neg_to_pos_mask = model.head.neg_to_pos_mask

        
    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        self.writer.add_scalar('l_pos_length', self.l_pos_length, self.get_iter(runner))
        self.writer.add_scalar('l_neg_length', self.l_neg_length, self.get_iter(runner))
        if hasattr(self, 'del_neg_mask') and self.del_neg_mask is not None:
            self.writer.add_image('del_neg_mask', self.del_neg_mask, global_step=1)
        if hasattr(self, 'neg_to_pos_mask') and self.neg_to_pos_mask is not None:
            self.writer.add_image('neg_to_pos_mask', self.neg_to_pos_mask, global_step=1)
    @master_only
    def after_run(self, runner) -> None:
        self.writer.close()