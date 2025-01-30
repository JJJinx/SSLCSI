import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
from mmcls.core import f1_score

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class CsiMultiViewDataset(BaseDataset):
    """The dataset outputs one view of an CSI sample, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipelines (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            #pipeline = Compose([build_from_cfg(pipe, PIPELINES)])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans  

    def __getitem__(self, idx):
        csi_record = self.data_source.get_csi(idx) # csi shape = [C,T] or [A,C,T] depends on the datasource parameter
        multi_views = list(map(lambda trans: trans(csi_record), self.trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(csi_record)) for csi_record in multi_views
            ]
        return dict(img=multi_views)

    def evaluate(self, results, logger=None):
        return NotImplemented