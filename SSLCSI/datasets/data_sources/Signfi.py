import os
import re
import csv
import os.path as osp
from unicodedata import name

import mmcv
import torch
import numpy as np
from pip import main

from ...builder import DATASOURCES
from ..base import BaseDataSource


@DATASOURCES.register_module()
class WiFi_Signfi_pt(BaseDataSource):
    #Actions are 276 gesture action
    def __init__(self, data_prefix,conj_pre,dual,keep_antenna=True):
        self.dataset = torch.load(data_prefix)
        super().__init__(data_prefix)
        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dual= dual

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            #info['gt_label'] = torch.tensor(gt_label.item(),dtype = torch.long)
            info['gt_label'] = torch.tensor(gt_label-1,dtype = torch.long) # original label has the range of 1-276
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        # input data has the shape of 
        # TODO what is the input data?
        out = self.dataset[idx][0][:,:,:,0].squeeze()
        out = out.reshape(out.shape[0],-1).permute(1,0) # permute to [C,T]
        if self.keep_antenna:
            out = out.view(3,out.size(0)//3,out.size(1))
        return out

@DATASOURCES.register_module()
class WiFi_Signfi_amp_pt(BaseDataSource):
    #Actions are 276 gesture action
    #def __init__(self, data_prefix,keep_antenna=True, classes=None, ann_file=None, test_mode=False, color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
    def __init__(self, data_prefix,keep_antenna=True, classes=None):

        self.dataset = torch.load(data_prefix)
        #super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)
        super().__init__(data_prefix, classes)
        self.keep_antenna = keep_antenna

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            #info['gt_label'] = torch.tensor(gt_label.item(),dtype = torch.long)
            info['gt_label'] = torch.tensor(gt_label-1,dtype = torch.long) # original label has the range of 1-276
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        out = self.dataset[idx][0][:,:,:,0].squeeze()
        out = out.reshape(out.shape[0],-1).permute(1,0) # permute to [C,T]
        if self.keep_antenna:
            out = out.view(3,out.size(0)//3,out.size(1))
        return out