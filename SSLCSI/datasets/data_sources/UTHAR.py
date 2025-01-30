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

def get_samples(root,act_dict,frame_act_dict):
    """Find samples by name under a root.

    Args:
        root (string): root directory of folders

    Returns:
        sample_list: list whose elements have form of ((action of the sample, action per frame), input file name)
    """
    file_list = os.listdir(root)
    annotation_list = []
    input_list = []
    for f in file_list:
        #if re.match('annotation', f):
        #    sample_action = [act_dict[action] for action in act_dict.keys() if action in f.split("_")][0]
        #    frame_action = np.loadtxt(osp.join(root,f),dtype=str)
        #    frame_action = np.array(list(map(lambda x:frame_act_dict[x],frame_action)))
        #    annotation_list.append((sample_action,frame_action))
        if re.match('input',f):
            input_list.append(f)
            sample_action = [act_dict[action] for action in act_dict.keys() if action in f.split("_")][0]
            annfile_name = f.replace('input','annotation')
            frame_action = np.loadtxt(osp.join(root,annfile_name),dtype=str)
            frame_action = np.array(list(map(lambda x:frame_act_dict[x],frame_action)))
            annotation_list.append((sample_action,frame_action))

    assert len(annotation_list) == len(input_list)
    sample_list = zip(annotation_list,input_list)
    return sample_list

def average_list(d_list):
    '''
    Args:
        d_list (list): shape [T,90]

    '''
    sum = [0.0 for _ in range(len(d_list[0]))] # for each channel
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum

def merge_timestamp(data, time_stamp, new_length = 2000):
    """align each samples time length

    Args:
        data (list): input signal list with shape [T,90]
        time_stamp (list): corresponding time stamp 

    Returns:
        aligned_data: list whose elements have the same length
    """
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / new_length
    cur_range = time_stamp[0] + intervel
    temp_list = []
    align_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                align_data.append(average_list(temp_list))
            else:
                align_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        align_data.append(average_list(temp_list))
    if len(align_data) < new_length:
        align_data.append(data[len(time_stamp)-1])
        print("shorter than new_length, add the last element")
    return align_data[:new_length]


@DATASOURCES.register_module()
class WiFi_Office(BaseDataSource):
    ACTIONS = {'bed':0,'run':1,'fall':2,'walk':3,'standup':4,'pickup':5,'sitdown':6}
    FRAME_ACTIONS = {'bed':0,'run':1,'fall':2,'walk':3,'standup':4,'pickup':5, 'sitdown':6,'NoActivity':7}
    def load_annotations(self):
        self.samples = get_samples(self.data_prefix,WiFi_Office.ACTIONS,WiFi_Office.FRAME_ACTIONS)
        data_infos = []
        for i,((gt_label,gt_frame_label),input_file_name) in enumerate(self.samples):
            info = {'csi_prefix': self.data_prefix}
            info['csi_info'] = {'filename': input_file_name}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['gt_frame_label'] = gt_frame_label
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        """Get CSI sample by index.

        Args:
            idx (int): Index of data.

        Returns:
            record: CSI signal shape = [C times T].
            time_stamp: time stamp shape = [T]
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if self.data_infos[idx].get('csi_prefix', None) is not None:
            if self.data_infos[idx]['csi_prefix'] is not None:
                filename = osp.join(
                    self.data_infos[idx]['csi_prefix'],
                    self.data_infos[idx]['csi_info']['filename'])
            else:
                filename = self.data_infos[idx]['csi_info']['filename']
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            record = []
            time_stamp = []
            for r in reader:
                record.append([float(str_d) for str_d in r[1:91]])
                time_stamp.append(float(r[0]))
            record = merge_timestamp(record, time_stamp)
            record = torch.tensor(record, dtype=torch.float32, requires_grad=False).t()
            #time_stamp = torch.tensor(time_stamp,dtype=torch.float32, requires_grad=False)
        out = record
        return out

@DATASOURCES.register_module()
class WiFi_Office_pt(BaseDataSource):
    ACTIONS = {'bed':0,'run':1,'fall':2,'walk':3,'standup':4,'pickup':5,'sitdown':6}
    FRAME_ACTIONS = {'bed':0,'run':1,'fall':2,'walk':3,'standup':4,'pickup':5, 'sitdown':6,'NoActivity':7}
    def __init__(self, data_prefix,keep_antenna=True, classes=None, ann_file=None, test_mode=False, color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
        self.dataset = torch.load(data_prefix)
        super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)
        self.keep_antenna = keep_antenna

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            #info['gt_label'] = torch.tensor(gt_label.item(),dtype = torch.long)
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        out = self.dataset[idx][0].squeeze().permute(1,0) # permute to [C,T]
        if self.keep_antenna:
            out = out.view(3,out.size(0)//3,out.size(1))
        return out