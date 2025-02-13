# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision.transforms.functional as TF
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose, RandomCrop

from .base import BaseDataset
from .builder import DATASETS, PIPELINES
from .utils import to_numpy


def image_to_patches(img):
    """Crop split_per_side x split_per_side patches from input image.

    Args:
        img (PIL Image): input image.

    Returns:
        list[PIL Image]: A list of cropped patches.
    """
    split_per_side = 3  # split of patches per image side
    patch_jitter = 21  # jitter of each patch from each grid
    h, w = img.size
    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    assert h_patch > 0 and w_patch > 0
    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            p = TF.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
            p = RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    return patches


@DATASETS.register_module()
class RelativeLocDataset(BaseDataset):
    """Dataset for relative patch location.

    The dataset crops image into several patches and concatenates every
    surrounding patch with center one. Finally it also outputs corresponding
    labels `0, 1, 2, 3, 4, 5, 6, 7`.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        format_pipeline (list[dict]): A list of dict, it converts input format
            from PIL.Image to Tensor. The operation is defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, format_pipeline, prefetch=False):
        super(RelativeLocDataset, self).__init__(data_source, pipeline,
                                                 prefetch)
        format_pipeline = [
            build_from_cfg(p, PIPELINES) for p in format_pipeline
        ]
        self.format_pipeline = Compose(format_pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = self.pipeline(img)
        patches = image_to_patches(img)
        if self.prefetch:
            patches = [torch.from_numpy(to_numpy(p)) for p in patches]
        else:
            patches = [self.format_pipeline(p) for p in patches]
        perms = []
        # create a list of patch pairs
        [
            perms.append(torch.cat((patches[i], patches[4]), dim=0))
            for i in range(9) if i != 4
        ]
        # create corresponding labels for patch pairs
        patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        return dict(
            img=torch.stack(perms), patch_label=patch_labels)  # 8(2C)HW, 8

    def evaluate(self, results, logger=None):
        return NotImplemented
