import torch
import torchvision.transforms.functional as TF
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose, RandomCrop
from mmcls.core import f1_score


from .base import BaseDataset
from .builder import DATASETS, PIPELINES
from .utils import to_numpy


def csi_to_patches(csi):
    """Crop split_per_side x split_per_side patches from input image.

    Args:
        csi (tensor): input csi tensor.

    Returns:
        list[tensor]: A list of cropped patches.
    """
    split_per_side = 3  # split of patches per image side
    #patch_jitter = 21  # jitter of each patch from each grid
    h_patch_jitter = 1
    w_patch_jitter = 10
    c,h,w = csi.size()
    h,w = csi.size()[1],csi.size()[2]

    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - h_patch_jitter
    w_patch = w_grid - w_patch_jitter
    assert h_patch > 0 and w_patch > 0
    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            p = TF.crop(csi, i * h_grid, j * w_grid, h_grid, w_grid)
            p = RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    return patches

def csi_to_patches2(csi):
    """Crop 1 (channel dim) x split_per_side (time dim) patches from input CSI signal.

    Args:
        csi (tensor): input csi tensor.

    Returns:
        list[tensor]: A list of cropped patches.
    """
    split_per_side = 3  # split of patches per image side
    #patch_jitter = 21  # jitter of each patch from each grid
    h_patch_jitter = 0
    w_patch_jitter = 10
    c,h,w = csi.size()
    h,w = csi.size()[1],csi.size()[2]

    h_grid = h // 3
    w_grid = w // split_per_side
    h_patch = h_grid - h_patch_jitter
    w_patch = w_grid - w_patch_jitter
    assert h_patch > 0 and w_patch > 0
    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            p = TF.crop(csi, i * h_grid, j * w_grid, h_grid, w_grid)
            p = RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    return patches


@DATASETS.register_module()
class RelativeLocCSIDataset(BaseDataset):
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

    def __init__(self, data_source, pipeline, format_pipeline, prefetch=False,csi2pat_func=False):
        super(RelativeLocCSIDataset, self).__init__(data_source, pipeline,
                                                 prefetch)
        format_pipeline = [
            build_from_cfg(p, PIPELINES) for p in format_pipeline
        ]
        self.format_pipeline = Compose(format_pipeline)
        self.csi2pat_func = csi2pat_func

    def __getitem__(self,idx):
        csi = self.data_source.get_csi(idx) # [A,C,T]  [2, 30, 500, 2]
        if csi.shape[0]<3:
            if len(csi.shape)==3:
                csi = csi.repeat(2,1,1)[:3,:,:] # [3,C,T]
                csi = self.pipeline(csi)
                csi = csi[:2,:,:] #[2,ci,ti]
            else:  # for dual situation
                csi = csi.repeat(2,1,1,1)[:3,:,:,:] # [3,C,T,2]
                csi_amp = csi[:,:,:,0]
                csi_conjang = csi[:,:,:,0]
                csi_amp = self.pipeline(csi_amp)[:2,:,:]
                csi_conjang = self.pipeline(csi_conjang)[:2,:,:]
                csi = torch.stack((csi_amp,csi_conjang),dim=3)
        else:
            csi = self.pipeline(csi) # (3,ci,ti) 经过剪裁以后的效果
        if self.csi2pat_func:
            patches = csi_to_patches2(csi)
            patch_labels = torch.LongTensor([0, 1])
            # if self.prefetch:
            #     patches = [torch.from_numpy(to_numpy(p)) for p in patches]
            # else:
            #     patches = [self.format_pipeline(p) for p in patches]
            perms = []
            # create a list of patch pairs
            [
                perms.append(torch.cat((patches[i], patches[1]), dim=0))
                for i in range(3) if i != 1
            ]
            # create corresponding labels for patch pairs
        else:
            patches = csi_to_patches(csi) #patch size [3, 13, 123]
            patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
            # if self.prefetch:
            #     patches = [torch.from_numpy(to_numpy(p)) for p in patches]
            # else:
            #     patches = [self.format_pipeline(p) for p in patches]
            perms = []
            # create a list of patch pairs
            [
                perms.append(torch.cat((patches[i], patches[4]), dim=0))
                for i in range(9) if i != 4
            ]
            # create corresponding labels for patch pairs
            
        return dict(
            img=torch.stack(perms), patch_label=patch_labels)  # 8(2A)CT/ [8, 6, 13, 123] 6表示2个patch的A

    def evaluate(self, results, logger=None):
        return NotImplemented
