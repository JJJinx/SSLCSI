from multiprocessing.pool import RUN
import os
import re
import csv
import random
import os.path as osp
from unicodedata import name

import pickle
import mmcv
import torch
import numpy as np
from pip import main

from ...builder import DATASOURCES
from ..base import BaseDataSource
from scipy import interpolate
from scipy.interpolate import interp1d


def align_time_length(signal,time_stamp,newtimelen):
    """
    One principle followed during the process of time-axis alignment 
    is to preserve as much information from the original data as possible.

    input shape [T,C,A]
    """
    if signal.shape[0]<newtimelen: # interpolation
        return updample(signal,time_stamp,newtimelen)
    if signal.shape[0]==newtimelen:
        return signal
    if signal.shape[0]>newtimelen: # using direct deletion followed by approximation of points
        return downsample(signal,time_stamp,newtimelen)

def updample(data,timestamp,exp_length):
    # input data shape [T,C,A]
    antenna_num = data.shape[-1]

    # generate the new time stamp
    interp_functions = [interp1d(timestamp, data[:, :, i], kind='linear', axis=0, fill_value='extrapolate') for i in range(antenna_num)]

    #interp_function = interp1d(timestamp,data, kind='linear', fill_value='extrapolate')

    new_timestamps = np.linspace(timestamp.min(), timestamp.max(), num=exp_length-len(timestamp))
    new_timestamps = np.concatenate((timestamp, new_timestamps))
    new_timestamps = np.sort(new_timestamps)
    interpolated_data = np.stack([interp(new_timestamps) for interp in interp_functions], axis=-1) # shape (T, C, A)
    return interpolated_data


def downsample(data,time_stamp,exp_length):
    #data shape should be [T,C,A]
    #Calculate new time intervals based on time stamps and the expected new time length.
    #Only one sample per time interval; achieved by taking individual point or averaging.

    #Calculate the average time interval for the new length, 
    #which is equivalent to averaging or using another method to combine the sample points within this time interval into a single point.
    intervel = (time_stamp[-1] - time_stamp[0]) / exp_length 
    cur_range = time_stamp[0] + intervel
    temp_list = []
    align_data = [data[0,:,:]] # init list with data at i=0
    nearest_i = -1

    for i in range(1,len(time_stamp)):  #Iterate through all sampling points.
        # If the current length of align_data reaches the last length, do not proceed to the next one directly; 
        #instead, combine all the remaining points into a single point.
        if len(align_data) == exp_length-1:
            align_data.append(data[-1,:,:])
            break
        #Count the elements between time[i] and time[i]+interval.
        if time_stamp[i]> len(align_data)*intervel: # If the time_stamp is outside the interval, then insert the nearest element to the interval.
            if len(temp_list) !=0:
                align_data.append(data[temp_list[-1],:,:])
            else:#If there are no values in the tmp list, insert the current value instead.
                align_data.append(data[i,:,:])
            temp_list=[]
        else:#If the time_stamp is within the interval, add the current point as the closest point.
            temp_list.append(i)

    if len(align_data) < exp_length:
        
        #Exclude the last element of align_data (to avoid duplication) and add the remaining elements.
        align_data = align_data[:-1]
        additional_number =  exp_length-len(align_data)
        tmp = data[len(time_stamp)-additional_number:,:,:]
        for i in range(additional_number):
            align_data.append(tmp[i,:,:])
        print("shorter than new_length, add the last element")
    align_data = np.stack(align_data)
    return align_data

def average_list(d_list):
    '''
    Args:
        d_list (list or array): length = T; element shape [C,A]

    '''
    if isinstance(d_list,list):
        d_list = np.stack(d_list)
    new_value = np.mean(d_list,axis=0)
    return new_value


@DATASOURCES.register_module()
class WiFi_Falldefi_pkl(BaseDataSource):
    def __init__(self, data_prefix,split_seed,mode,conj_pre,dual,prop=1.0,multi_dataset=False, classes=None,keep_antenna=True):
     #ann_file=None, test_mode=False, color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
        
        self.keep_antenna = keep_antenna
        f = open(data_prefix,'rb')
        pkl_data = pickle.load(f)
        self.csi  = pkl_data['record']       # a list of complex signal, length is equal to the number of samples
        self.label = pkl_data['label']       # a list, length is equal to the number of samples
        self.time_stamps = pkl_data['stamp'] # a list
        total_index = list(range(len(self.label)))

        self.conj_pre = conj_pre
        self.dual= dual

        # shuffle the index
        random.seed(split_seed)
        random.shuffle(total_index)
        # trainval/test split 8:2
        split_ratio = 0.8
        split_index = int(len(total_index) * split_ratio)
        trainval_index_list = total_index[:split_index]
        self.test_index = total_index[split_index:]
        # train/val split 9:1
        split_ratio = 0.9
        split_index = int(len(trainval_index_list) * split_ratio)
        self.train_index = trainval_index_list[:split_index]
        self.val_index = trainval_index_list[split_index:]

        if mode == 'train':
            self.csi  = [self.csi[i] for i in self.train_index]      # a list of complex signal, length is equal to the number of samples
            self.label = [self.label[i] for i in self.train_index]       # a list, length is equal to the number of samples
            self.time_stamps = [self.time_stamps[i] for i in self.train_index] # a list
        if mode == 'val':
            self.csi  = [self.csi[i] for i in self.val_index]      # a list of complex signal, length is equal to the number of samples
            self.label = [self.label[i] for i in self.val_index]       # a list, length is equal to the number of samples
            self.time_stamps = [self.time_stamps[i] for i in self.val_index] # a list
        if mode == 'test':
            self.csi  = [self.csi[i] for i in self.test_index]      # a list of complex signal, length is equal to the number of samples
            self.label = [self.label[i] for i in self.test_index]       # a list, length is equal to the number of samples
            self.time_stamps = [self.time_stamps[i] for i in self.test_index] # a list
        super().__init__(data_prefix, classes)#, ann_file, test_mode, color_type, channel_order, file_client_args)

    def load_annotations(self):
        data_infos = []
        for i,gt_label in enumerate(self.label):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos

    def unwrap(self,phase):
        return np.unwrap(phase)
    
    def conj_multi(self,csi):
        # csi shape = [T,C,A] [T,30,3]
        ref_ant = np.expand_dims(np.conjugate(csi[:,:,0]),axis=2)
        conjed_phase = np.angle(csi*ref_ant)[:,:,[1,2]]
        return conjed_phase

    def get_csi(self, idx):
        out = self.csi[idx].squeeze() # shape [T,C,A]
        stamp = self.time_stamps[idx].squeeze() # shape [T,]
        
        if self.dual == 'amp':
            out = np.abs(out)  # T,C,A
            out = torch.from_numpy(align_time_length(out,stamp,10000)) 
            out = out.permute(2,1,0) #permute to  [A,C,T]
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        
        if self.dual == 'pha':
            # conj(depends)>unwrap>aligntime
            if self.conj_pre == True: # conj
                out = self.conj_multi(out)
            out = self.unwrap(out) # unwrap
            out = torch.from_numpy(align_time_length(out,stamp,10000)) # align shape [T,C,A]
            out = out.permute(2,1,0) # permute to  [A,C,T]
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        
        if self.dual == 'sep': # shape should be [A,C,T,2]
            if self.conj_pre == True:
                amp = np.abs(out)[:,:,[1,2]] # [T,C,A]
                amp = torch.from_numpy(align_time_length(amp,stamp,10000))
                conj_phase = self.conj_multi(out)
                conj_phase = self.unwrap(conj_phase)
                conj_phase = torch.from_numpy(align_time_length(conj_phase,stamp,10000))
                out = torch.stack((amp,conj_phase),dim = -1)  # [T,C,A,2]
            else:
                amp = np.abs(out) # [T,C,A]
                amp = torch.from_numpy(align_time_length(amp,stamp,10000))
                phase = np.angle(out)
                phase = self.unwrap(phase)
                phase = torch.from_numpy(align_time_length(phase,stamp,10000))
                out = torch.stack((amp,phase),dim = -1) # [T,C,A,2]
            
            out = out.permute(2,1,0,3) # [A,C,T,2]
            if self.keep_antenna == False: # output shape should be [AC,T,2]
                time_length = out.shape[2]
                out = out.permute(2,3,0,1).reshape(time_length,2,-1).permute(2,0,1) #shape [AC,T,2]

        if self.dual == 'con':   # shape should be [A,2C,T]
            if self.conj_pre == True:
                amp = np.abs(out)[:,:,[1,2]] # [T,C,A]
                amp = torch.from_numpy(align_time_length(amp,stamp,10000))
                conj_phase = self.conj_multi(out)
                conj_phase = self.unwrap(conj_phase)
                conj_phase = torch.from_numpy(align_time_length(conj_phase,stamp,10000))
                out = torch.cat((amp,conj_phase),dim=1) # [T,2C,A]
            else:
                amp = np.abs(out) # [T,C,A]
                amp = torch.from_numpy(align_time_length(amp,stamp,10000))
                phase = np.angle(out) # [T,C,A]
                phase = self.unwrap(phase)
                phase = torch.from_numpy(align_time_length(phase,stamp,10000))
                out = torch.cat((amp,phase),dim =1) # [T,2C,A]

            out = out.permute(2,1,0) # [A,2C,T]
            if self.keep_antenna == False: # output shape should be [2AC,T]
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [2AC,T]
        return out.float()


        