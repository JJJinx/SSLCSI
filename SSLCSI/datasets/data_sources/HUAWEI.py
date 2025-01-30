from multiprocessing.pool import RUN
import os
import re
import csv
import os.path as osp
from unicodedata import name

import mmcv
import torch
import numpy as np
from pip import main
from torch.utils.data import ConcatDataset


from ....utils import wden_filter,butter_amp_filter,csi_segmentation
from ...builder import DATASOURCES
from ..base import BaseDataSource


@DATASOURCES.register_module()
class WiFi_Huawei_amp(BaseDataSource):
    def __init__(self, data_prefix_train,data_prefix_val,data_prefix_test,mode,data_prefix=None,
                 multi_dataset=False,keep_antenna=False,conj_pre=False,dual=False, classes=None, prop=1.0,
            ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # data_prefix is a list
        # train
        if mode == 'train':
            data_prefix = data_prefix_train
        if mode == 'val':
            data_prefix = data_prefix_val
        if mode == 'test':
            data_prefix = data_prefix_test
        set_list = []
        for train_path in data_prefix:
            filelist = os.listdir(train_path)
            for file in filelist:
                file_path = os.path.join(train_path,file)
                tmp = self.load_npy_to_tensor(file_path)
                set_list.append(tmp)
        self.dataset=ConcatDataset(set_list)
        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dual = dual
        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)
    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        amp = datadict.item()['amp']  # shape (N,T, 4, 64)
        #phase = datadict.item()['phase']
        # filter the useless subcarriers, and divide them into amp and phase
        amp = amp
        #phase = phase[:,:,:,4:]
        records =amp  #shape (N,T, 4, 60,2) (151, 300, 4, 60, 2)
        #labels = datadict.item()['labels']  # shape (N)
        #labels = np.array(labels>0,dtype=np.int32)  #only consider whether there are human 
        #assert records.shape[0]==labels.shape[0]
        records = torch.from_numpy(records).float()
        #labels = torch.from_numpy(labels)
        #dataset = torch.utils.data.TensorDataset(records, labels)
        dataset = torch.utils.data.TensorDataset(records)
        return dataset

    def load_annotations(self):
        data_infos = []
        for i,(feature,) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(i,dtype = torch.long)
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):

        out = self.dataset[idx][0]  #shape (T, 4, 64)
        #out shape [T,C,antenna,2(amp,phase)]
        out = out.permute(0,2,1)  # T C A

        if len(out.shape)==3:
            #out shape [T,C,antenna]
            if self.keep_antenna== True:
                out = out.permute(2,1,0) # permute to [A,C,T]
                return out
            out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
            out = out.permute(1,0) # permute to [C,T]
        return out        

@DATASOURCES.register_module()
class WiFi_Huawei_ang(BaseDataSource):
    def __init__(self, data_prefix_train,data_prefix_val,data_prefix_test,mode,data_prefix=None,
                 multi_dataset=False,keep_antenna=False,conj_pre=False,dual=False, classes=None, prop=1.0,
            ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train
        if mode == 'train':
            data_prefix = data_prefix_train
        if mode == 'val':
            data_prefix = data_prefix_val
        if mode == 'test':
            data_prefix = data_prefix_test
        set_list = []
        for train_path in data_prefix:
            filelist = os.listdir(train_path)
            for file in filelist:
                file_path = os.path.join(train_path,file)
                tmp = self.load_npy_to_tensor(file_path)
                set_list.append(tmp)
        self.dataset=ConcatDataset(set_list)

        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dual = dual

        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)
    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        amp = datadict.item()['amp']  
        phase = datadict.item()['phase'] # shape (N,T, 3, 64) (151, 100, 3, 64)
        amp = amp
        phase = phase
        records = phase  #shape (N,T, 3, 60) 
        #labels = datadict.item()['labels']  # shape (N)
        #labels = np.array(labels>0,dtype=np.int32)  
        #assert records.shape[0]==labels.shape[0]
        records = torch.from_numpy(records).float()
        #labels = torch.from_numpy(labels)
        #dataset = torch.utils.data.TensorDataset(records, labels)
        dataset = torch.utils.data.TensorDataset(records)
        return dataset

    def load_annotations(self):
        data_infos = []
        #for i,(feature,gt_label) in enumerate(self.dataset):
        for i,(feature,) in enumerate(self.dataset):
            info = {}
            #info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['gt_label'] = torch.tensor(i,dtype = torch.long)
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        out = self.dataset[idx][0]  #shape (T, 3, 64)
        #out shape [T,C,antenna]
        out = out.permute(0,2,1)  # T C A
        #out shape  [100, 60, 3]
        #out = out[:,:,:].squeeze()  # [T,C,antenna]
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out        

@DATASOURCES.register_module()
class WiFi_Huawei_ang_file(BaseDataSource):
    def __init__(self, data_prefix_train,data_prefix_val,data_prefix=None,num_detect=False,scene=None,
                 keep_antenna=False, classes=None,ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train
        self.num_detect = num_detect
        self.keep_antenna = keep_antenna
        if data_prefix_train is not None:
            data_prefix = data_prefix_train
            set_list = []
            for train_path in data_prefix:  
                filelist = os.listdir(train_path)
                for file in filelist:
                    file_path = os.path.join(train_path,file)
                    if file_path not in data_prefix_val:
                        tmp = self.load_npy_to_tensor(file_path)
                        set_list.append(tmp)
            self.dataset=ConcatDataset(set_list)
        else:
            set_list = []
            data_prefix = data_prefix_val
            for file_path in data_prefix:
                tmp = self.load_npy_to_tensor(file_path)
                set_list.append(tmp)
            self.dataset=ConcatDataset(set_list)
        

        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)
    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        amp = datadict.item()['amp']  
        phase = datadict.item()['phase'] # shape (N,T, 3, 64) (151, 100, 3, 64)
        records = phase  #shape (N,T, 3, 60) 
        labels = datadict.item()['labels']  # shape (N)
        records = torch.from_numpy(records).float()
        if self.num_detect==False:   ##whether consider number of persons detection
            labels = np.array(labels>0,dtype=np.int32)  
            labels = torch.from_numpy(labels)
        else:
            labels = torch.from_numpy(labels) # shape N_i
            non_zero_idx = labels.nonzero().squeeze()
            records = records[non_zero_idx,:,:,:]
            labels = labels[non_zero_idx]-1 # Preventing the label array from going out of bounds.
        dataset = torch.utils.data.TensorDataset(records, labels)
        return dataset

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        out = self.dataset[idx][0]  #shape (T, 3, 64)
        #out shape [T,C,antenna]
        out = out.permute(0,2,1)  # T C A
        #out shape  [100, 60, 3]
        #out = out[:,:,:].squeeze()  # [T,C,antenna]
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out    

@DATASOURCES.register_module()
class WiFi_Huawei_ang_pt(BaseDataSource):
    def __init__(self,data_prefix=None,multi_dataset=False,keep_antenna=False,scene='office',num_detect=False, 
                 classes=None,ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train
        self.keep_antenna = keep_antenna
        self.scene = scene
        self.num_detect = num_detect 
        self.dataset = self.load_npy_to_tensor(data_prefix)

        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)

    def load_annotations(self):
        data_infos = []
        if self.scene == 'home':
            records,labels,living_label =self.dataset
            num_data = records.shape[0]
            for i in range(num_data):
                info = {}
                info['gt_label'] = torch.tensor(labels[i],dtype = torch.long)
                info['living_label'] = torch.tensor(living_label[i],dtype = torch.long)
                info['idx'] = int(i)
                data_infos.append(info)

        else:
            records,labels =self.dataset  #torch.Size([9840, 100, 3, 60]) torch.Size([9840])
            num_data = records.shape[0]
            for i in range(num_data):
                info = {}
                info['gt_label'] = torch.tensor(labels[i],dtype = torch.long)
                info['idx'] = int(i)
                data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        if self.scene == 'home':
            records,labels,living_label =self.dataset
        else:
            records,labels =self.dataset
        out = records[idx]  #shape (T, 3, 60) torch.Size([100, 3, 60])
        #out shape [T,C,antenna]
        out = out.permute(0,2,1)  # T C A
        #out shape  [100, 60, 3]
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out  

    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        datadict = datadict.item()
        amp = datadict['amp']  # shape (N,T, 4, 60)
        phase = datadict['phase'] # shape (N,T, 3, 60)
        records =phase  #shape (N,T, 4, 60) (151, 300, 4, 60, 2)
        records = torch.from_numpy(records).float() #shape (N,T, 3, 60)
        labels = datadict['label'] #shape (N,) <class 'numpy.ndarray'>
        if self.num_detect==False:   
            labels = labels>0
            labels = torch.from_numpy(labels)
            if self.scene == 'home':
                living_label = datadict['liviing']  #shape (N,)
                living_label = living_label>0
                dataset = (records,labels,living_label)
            else:
                dataset = (records,labels)
        else: 
            labels = torch.from_numpy(labels) # shape N_i
            non_zero_idx = labels.nonzero().squeeze()
            records = records[non_zero_idx,:,:,:]
            labels = labels[non_zero_idx]-1 
            if self.scene == 'home':
                living_label = datadict['liviing']
                dataset = (records,labels,living_label)
            else:
                dataset = (records,labels)
        return dataset

@DATASOURCES.register_module()
class WiFi_Huawei_home_ang_pt(BaseDataSource):
    def __init__(self,data_prefix=None,multi_dataset=False,keep_antenna=False,scene='office',num_detect=False, 
                 classes=None,ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train
        self.keep_antenna = keep_antenna
        self.scene = scene
        self.num_detect = num_detect 
        self.dataset = self.load_npy_to_tensor(data_prefix)

        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)

    def load_annotations(self):
        data_infos = []
        if self.scene == 'home':
            records,labels,living_label =self.dataset
            num_data = records.shape[0]
            for i in range(num_data):
                info = {}
                info['gt_label'] = torch.tensor(labels[i],dtype = torch.long)
                info['living_label'] = torch.tensor(living_label[i],dtype = torch.long)
                info['idx'] = int(i)
                data_infos.append(info)

        else:
            records,labels =self.dataset  #torch.Size([9840, 100, 3, 60]) torch.Size([9840])
            num_data = records.shape[0]
            for i in range(num_data):
                info = {}
                info['gt_label'] = torch.tensor(labels[i],dtype = torch.long)
                info['idx'] = int(i)
                data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        if self.scene == 'home':
            records,labels,living_label =self.dataset
        else:
            records,labels =self.dataset
        out = records[idx]  #shape (T, 3, 60) torch.Size([100, 3, 60])
        #out shape [T,C,antenna]
        out = out.permute(0,2,1)  # T C A
        #out shape  [100, 60, 3]
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out  

    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        datadict = datadict.item()
        amp = datadict['amp']  # shape (N,T, 4, 60)
        phase = datadict['phase'] # shape (N,T, 3, 60)
        records =phase  #shape (N,T, 4, 60) (151, 300, 4, 60, 2)
        records = torch.from_numpy(records).float() #shape (N,T, 3, 60)
        labels = datadict['label'] #shape (N,) <class 'numpy.ndarray'>
        if self.num_detect==False:   
            labels = labels>0
            labels = torch.from_numpy(labels)
            if self.scene == 'home':
                living_label = datadict['liviing']  #shape (N,)
                living_label = living_label>0
                dataset = (records,labels,living_label)
            else:
                dataset = (records,labels)
        else: # If considering the number of people, only train the classifier using samples with a non-zero number of people.
            labels = torch.from_numpy(labels) # shape N_i
            non_zero_idx = labels.nonzero()
            records = records[non_zero_idx,:,:,:]
            labels = labels[non_zero_idx]
            if self.scene == 'home':
                living_label = datadict['liviing']
                living_label = living_label[non_zero_idx]
                dataset = (records,labels,living_label)

            else:
                dataset = (records,labels)

        return dataset

@DATASOURCES.register_module()
class WiFi_Huawei_score(BaseDataSource):
    def __init__(self,data_prefix=None,keep_antenna=False, classes=None, prop=1.0,
                ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train
        self.dataset = self.load_npy_to_tensor(data_prefix)
        self.keep_antenna = keep_antenna
        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)

    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        amp = datadict.item()['amp']  
        phase = datadict.item()['phase'] # shape (N,T, 3, 64) (151, 100, 3, 64)
        amp = amp
        phase = phase
        records = phase  #shape (N,T, 3, 60) 
        records = torch.from_numpy(records).float()
        dataset = torch.utils.data.TensorDataset(records)
        return dataset
    def load_annotations(self):
        data_infos = []
        #for i,(feature,gt_label) in enumerate(self.dataset):
        for i,(feature,) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(i,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos

    def get_csi(self, idx):
        out = self.dataset[idx][0]  #shape (T, 3, 64)
        out = out.permute(0,2,1)  # T C A
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out    

@DATASOURCES.register_module()
class WiFi_Huawei_livingroom(BaseDataSource):
    def __init__(self,data_prefix=None,keep_antenna=False, classes=None, prop=1.0,
                ann_file=None, test_mode=False, file_client_args=dict(backend='disk')):
        # train        
        set_list = []
        for file_path in data_prefix:
            tmp = self.load_npy_to_tensor(file_path)
            set_list.append(tmp)
        self.dataset=ConcatDataset(set_list)
        self.keep_antenna = keep_antenna
        super().__init__(data_prefix, classes, ann_file, test_mode, file_client_args)

    def load_npy_to_tensor(self,file_path):
        datadict = np.load(file_path,allow_pickle=True)
        datadict = datadict.item()
        amp = datadict['amp']  
        phase = datadict['phase'] # shape (N,T, 3, 64) (151, 100, 3, 64)
        amp = amp
        phase = phase
        records = phase  #shape (N,T, 3, 60) 
        records = torch.from_numpy(records).float()
        if 'living_labels' in datadict.keys():
            living_label = datadict['living_labels']
            living_label = torch.from_numpy(living_label)
        else:
            num_sample = amp.shape[0]
            living_label = -1*torch.ones(num_sample)
        dataset = torch.utils.data.TensorDataset(records,living_label)
        return dataset
    
    def load_annotations(self):
        data_infos = []
        for i,(feature,living_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(living_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos

    def get_csi(self, idx):
        out = self.dataset[idx][0]  #shape (T, 3, 64)
        out = out.permute(0,2,1)  # T C A
        if self.keep_antenna== True:
            out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        out = out.reshape(out.shape[0],-1) #[T,CA] on cpu
        out = out.permute(1,0) # permute to [C,T]
        return out    
