import torch
from mmcv.utils import print_log
from mmcls.core import f1_score
from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy

from sklearn.metrics import confusion_matrix

@DATASETS.register_module()
class CsiSingleViewDataset_Huawei(BaseDataset):
    """The dataset outputs one view of an csi record, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(CsiSingleViewDataset_Huawei, self).__init__(data_source, pipeline,
                                                prefetch)
        self.gt_labels = self.data_source.get_gt_labels()

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        csi = self.data_source.get_csi(idx)
        csi = self.pipeline(csi)
        if self.prefetch:
            csi = torch.from_numpy(to_numpy(csi))
        return dict(img=[csi], label=label, idx=idx) #keep the key name unchange

    def evaluate(self, results, logger=None, topk=(1,)):
        """
        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values. shape [len(dataloader),classes], it is the output of the head
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}
        for name, val in results.items(): #name is the name of the head
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_gt_labels()) # shape [len(dataloader)]
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')
            num = val.size(0)
            _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True) # pred is the predicted class not the score
            pred = pred.t() # shape [k,len(dataloader)] since here we only consider the existence of human , k=1
            # calculate confusion matrix
            # cm = confusion_matrix(target, pred.squeeze())
            # tn, fp, fn, tp = cm.ravel()
            # tn, fp, fn, tp = tn.item(), fp.item(), fn.item(), tp.item()
            # Pd = tp/(tp+fn)
            # Pfa = fp/(fp+tn)
            # score = Pd*target.sum()/len(target)+(1-Pfa)*(len(target)-target.sum())/len(target)
            # score = score.item()
            # k=1
            # eval_res[f'{name}_top{k}'] = score
            # if logger is not None and logger != 'silent':
            #         print_log(f'{name}_top{k}: {score:.03f},{name}_tp: {tp:d}, {name}_fp: {fp:d},{name}_tn: {tn:d}, {name}_fn: {fn:d}', logger=logger)
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                f1 = f1_score(val,target) 
                eval_res[f'{name}_top{k}'] = f1
                acc = correct_k * 100.0 / num
                eval_res[f'{name}_acc'] = acc
                if logger is not None and logger != 'silent':
                    print_log(f'{name}_top{k}: {f1:.03f}, {name}_acc: {acc:.03f}', logger=logger)
        return eval_res







@DATASETS.register_module()
class CsiSingleViewUDADataset_Huawei(BaseDataset):
    """The dataset outputs one view of an csi record, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(CsiSingleViewUDADataset_Huawei, self).__init__(data_source, pipeline,
                                                prefetch)
        self.gt_labels = self.data_source.get_gt_labels()

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        csi = self.data_source.get_csi(idx)
        csi = self.pipeline(csi)
        if self.prefetch:
            csi = torch.from_numpy(to_numpy(csi))
        return dict(img=[csi], label=label, idx=idx) #keep the key name unchange

    def evaluate(self, results, logger=None, topk=(1,)):
        """
        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values. shape [len(dataloader),classes], it is the output of the head
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}
        for name, val in results.items(): #name is the name of the head
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_test_target()) # shape [len(dataloader)]
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')
            num = val.size(0)
            _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True) # pred is the predicted class not the score
            pred = pred.t() # shape [k,len(dataloader)] since here we only consider the existence of human , k=1
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                f1 = f1_score(val,target) 
                eval_res[f'{name}_top{k}'] = f1
                acc = correct_k * 100.0 / num
                eval_res[f'{name}_acc'] = acc
                if logger is not None and logger != 'silent':
                    print_log(f'{name}_top{k}: {f1:.03f}, {name}_acc: {acc:.03f}', logger=logger)
        return eval_res