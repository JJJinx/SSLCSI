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
        data (ndarray): input signal list with shape [T,90]
        time_stamp (list): corresponding time stamp 

    Returns:
        aligned_data(list): list whose elements have the same length
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


def widar_data_info(file_name):
    data_info = file_name.split('-')
    user_id = data_info[0]
    gesture = data_info[1]
    torso_location = data_info[2]
    face_orientation = data_info[3]
    repeat_index = data_info[4]
    rx_id = data_info[5].split('.')[0]
    data_info = {'user':user_id,
                 'gesture':gesture,
                 'torso_loc':torso_location,
                 'face_ori':face_orientation,
                 'repeat_ind':repeat_index,
                 'rx':rx_id
    }
    return data_info


def get_samples(file_list):
    """Find samples by name under a root.
    currently only consider the situation of single rx. TODO

    Args:
        file_list (ndarray): an array of file name

    Returns:
        sample_list: list whose elements have form of ((action of the sample, action per frame), input file name)
    """
    annotation_list = []
    input_list = []
    for f in file_list:
        data_info = widar_data_info(f)
        if data_info['rx'] != 'r1': # currently only consider the single rx situation TODO
            continue
        annotation_list.append(int(data_info['gesture'])-1) # from 0 to 5
        input_list.append(f)
    assert len(annotation_list) == len(input_list)
    sample_list = zip(annotation_list,input_list)
    return sample_list


@DATASOURCES.register_module()
class WiFi_Widar(BaseDataSource):
    """
        directly read widar data from .dat files
    """
    def __init__(self):
        super().__init__()

    def load_annotations(self):
        self.mode = self.data_prefix.split('-')[-1]
        data_prefix = self.data_prefix.split('-')[0]
        if self.mode == 'train':
            # based on /media/mnt/056EB4A96239713D/xuke_arap/widar/CSI_20181109/20181109/user1
            file_list = '/media/mnt/056EB4A96239713D/xuke_arap/widar/widar_train.txt'
        if self.mode == 'val':
            file_list = '/media/mnt/056EB4A96239713D/xuke_arap/widar/widar_val.txt'
        if self.mode == 'test':
            file_list = '/media/mnt/056EB4A96239713D/xuke_arap/widar/widar_test.txt'
        file_list = np.loadtxt(file_list,dtype=str)
        self.samples = get_samples(file_list)
        data_infos = []
        for i,(gt_label,input_file_name) in enumerate(self.samples):
            info = {'csi_prefix': data_prefix}
            info['csi_info'] = {'filename': input_file_name}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
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
        # load the dat file 
        csidata = Intel(filename, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
        csidata.read()
        record = csidata.get_scaled_csi() # shape [T,30,3,1]
        record = np.abs(record) 
        record = record.transpose(0,2,1,3).reshape(record.shape[0],-1) 
        time_stamp = csidata.timestamp_low # list length = [T]
        record = np.array(merge_timestamp(record, time_stamp,new_length=800)) # a list 
        record = torch.tensor(record, dtype=torch.float32, requires_grad=False).t()  
        #time_stamp = torch.tensor(time_stamp,dtype=torch.float32, requires_grad=False)
        out = record
        return out

@DATASOURCES.register_module()
class Widar_pt(BaseDataSource):
    def __init__(self, data_prefix, data_prefix_train=None, data_prefix_val=None, data_prefix_test=None,
                 multi_dataset=False,prop=1.0,keep_antenna=False,conj_pre=False,dual='amp'):
        """
            data prefix augments is used for locating the data files

        """
        if multi_dataset:
            self.dataset = ConcatDataset([torch.load(data_prefix_train),
                           torch.load(data_prefix_val),
                           torch.load(data_prefix_test)])
        else:    
            self.dataset = torch.load(data_prefix)
        self.prop = prop
        train_len = int(prop*len(self.dataset))
        res_len = len(self.dataset)-train_len 
        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dual = dual

        self.dataset, _ = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_len,res_len],
            generator=torch.Generator().manual_seed(0)
        )
        super().__init__(data_prefix)

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    


    def get_csi(self, idx):
        out = self.dataset[idx][0]  
        #out shape [T,C,antenna,4(amp,phase,ratio_amp，ratio_ang)]
        if self.dual == 'amp':
            out = out[:,:,:,0].squeeze()  # [T,C,antenna]
            if self.keep_antenna== False:
                time_length = out.shape[0]
                out = out.reshape(time_length,-1).permute(1,0) #shape [AC,T]
            else:
                out = out.permute(2,1,0) # permute to [A,C,T]
            return out
        
        if self.dual == 'pha':
            # conj(depends)>unwrap>aligntime
            if self.conj_pre == True: # do the conj mul to the csi data
                #out shape [T,C,antenna,4(amp,ang,ratio_amp，ratio_ang)]
                out = out[:,:,:,3].squeeze()  # [T,C,antenna]
                out = out[:,:,[0,2]]  # [T,C,antenna]  # remove the reference antenna
            else:
                out = out[:,:,:,1].squeeze()  # [T,C,antenna]
            # format the out shape as [A,C,T ] or [AC,T]
            if self.keep_antenna== False:
                time_length = out.shape[0]
                out = out.reshape(time_length,-1).permute(1,0) #shape [AC,T]
            else:
                out = out.permute(2,1,0) # permute to [A,C,T]
            return out
                
        if self.dual == 'sep': # shape should be [A,C,T,2]
        #out shape [T,C,antenna,4(amp,phase,ratio_amp，ratio_ang)]
            if self.conj_pre == True: # do the conj mul to the csi data
                out = out[:,:,:,[0,3]]  # choose amp and ratio_ang
                out = out[:,:,[0,2],:] # choose antennas [T,C,antenna,2(amp,ratio_ang)]
            else: # no conj
                out = out[:,:,:,[0,1]]  #  [T,C,antenna,2(amp,phase)]
            # format the output as [AC,T,2] or [A,C,T,2]
            if self.keep_antenna==False:  # output [AC T 2]
                    time_length = out.shape[0]
                    out = out.permute(2,3,0,1) # [T,2(amp，ratio_ang),antenna,C]
                    out = out.reshape(out.shape[0],out.shape[1],-1) #[T,2(amp，ratio_ang),AC] on cpu
                    out = out.permute(2,0,1) # permute to [C,T,2]
            else: # output [A C T 2]
                out = out.permute(2,1,0,3)# [antenna,C,T,2(amp，ratio_ang)]
            return out
        
        if self.dual == 'con':   # shape should be [A,2C,T]
            if self.conj_pre == True:
                out = out[:,:,:,[0,3]] 
                out = out[:,:,[0,2],:]  #[T,C,2(a0,a2),2(amp，ratio_ang)]
                out = out.permute(0,2,1,3).reshape(out.shape[0],2,-1) #[T,A(=2),2C]
            else:
                out = out[:,:,:,[0,1]] #[T,C,A,2(amp，phase)]
                out = out.permute(0,2,1,3).reshape(out.shape[0],2,-1) #[T,A(=2),2C]
            if self.keep_antenna == False: # output shape should be [2AC,T]
                out = out.reshape(out.shape[0],-1) #[T,2AC] on cpu
                out = out.permute(1,0) # permute to [2AC,T]
            else:
                out = out.permute(1,2,0) # permute to [A,2C,T]
            return out
##如果上面的这个好用就可以删除下面的这些
#####################################
@DATASOURCES.register_module()
class WiFi_Widar_pt(BaseDataSource):
    def __init__(self, data_prefix, data_prefix_train=None, data_prefix_val=None, data_prefix_test=None,
                 multi_dataset=False,keep_antenna=False,conj_pre=False,dual=False, classes=None, prop=1.0,
            ann_file=None, test_mode=False, color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
        if multi_dataset:
            self.dataset = ConcatDataset([torch.load(data_prefix_train),
                           torch.load(data_prefix_val),
                           torch.load(data_prefix_test)])
        else:    
            self.dataset = torch.load(data_prefix)
        self.prop = prop
        train_len = int(prop*len(self.dataset))
        res_len = len(self.dataset)-train_len 
        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dual = dual

        self.dataset, _ = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_len,res_len],
            generator=torch.Generator().manual_seed(0)
        )
        super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)


    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = gt_label.clone().detach().long()
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):

        out = self.dataset[idx][0]  
        if len(out.shape)==3:
            # out shape should be [T,C,2(amp,phase)]
            print(out.shape)
            raise RuntimeError
            out = torch.tensor(out,dtype=torch.float)
            out = out.reshape(out.shape[0],-1).permute(1,0)   
        if len(out.shape)==4: #[500, 30, 3, 4] [T,C,A,4]
            if self.conj_pre == True: # do the conj mul to the csi data
                #out shape [T,C,antenna,4(amp,phase,ratio_amp，ratio_ang)]
                if self.dual == True: #Process phase and amplitude separately
                    out = out[:,:,:,[0,3]] 
                    out = out[:,:,[0,2],:]
                    if self.keep_antenna==True:
                        out = out.permute(2,1,0,3)# [antenna,C,T,2(amp，ratio_ang)]
                        return out
                    out = out.permute(2,3,0,1) # [T,2(amp，ratio_ang),antenna,C]
                    out = out.reshape(out.shape[0],out.shape[1],-1) #[T,2(amp，ratio_ang),AC] on cpu
                    out = out.permute(2,0,1) # permute to [C,T,2]
                    return out
                out = out[:,:,:,[0,3]] 
                out = out[:,:,[0,2],:]  #[T,C,2(a0,a2),2(amp，ratio_ang)]
                out = out.permute(0,2,1,3).reshape(out.shape[0],2,-1) #[T,A(=2),2C]
                if self.keep_antenna== True:
                    out = out.permute(1,2,0) # permute to [A,2C,T]
                    return out
                out = out.reshape(out.shape[0],-1) #[T,2AC] on cpu
                out = out.permute(1,0) # permute to [C,T]
                return out

            #out shape [T,C,antenna,2(amp,phase)]
            out = out[:,:,:,[0,1]] 
            out = out.permute(0,2,1,3) #[T,A,C,2(amp,phase)] arrange the data along the C dimension during reshaping.
            out = out.reshape(out.shape[0],out.shape[1],-1) #[T,A,2C]
            if self.keep_antenna== True:
                out = out.permute(1,2,0) # permute to [A,C,T]
                return out
            out = out.reshape(out.shape[0],-1) #[T,2AC] on cpu
            out = out.permute(1,0) # permute to [C,T]
        return out        


@DATASOURCES.register_module()
class WiFi_Widar_amp_pt(BaseDataSource):
    def __init__(self, data_prefix, data_prefix_train=None, data_prefix_val=None, data_prefix_test=None,
                  multi_dataset=False,keep_antenna=False,conj_pre=False,dual=False, classes=None,prop=1.0,
                    ann_file=None, test_mode=False,color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
        if multi_dataset:
            self.dataset = ConcatDataset([torch.load(data_prefix_train),
                           torch.load(data_prefix_val),
                           torch.load(data_prefix_test)])
        else:    
            self.dataset = torch.load(data_prefix)
        self.prop = prop
        train_len = int(prop*len(self.dataset))
        res_len = len(self.dataset)-train_len 
        self.keep_antenna = keep_antenna

        self.dataset, _ = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_len,res_len],
            generator=torch.Generator().manual_seed(0)
        )
        super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)
        self.denoising_flag = False
    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos

    def get_csi(self, idx):
        out = self.dataset[idx][0]
        if len(out.shape)==3:
            print(out.shape)
            raise RuntimeError
            out = torch.tensor(out,dtype=torch.float)
            out = out.reshape(out.shape[0],-1).permute(1,0)
        if len(out.shape)==4:
            #out shape [T,C,antenna,2(amp,phase)]
            out = out[:,:,:,0].squeeze()  # [T,C,antenna]
            if self.keep_antenna== True:
                out = out.permute(2,1,0) # permute to [A,C,T]
                return out

            out = out.reshape(out.shape[0],-1) #[T,C] on cpu
            if self.denoising_flag==True:
                out = torch.tensor(self.denoising(out),dtype=torch.float)
            out = out.permute(1,0) # permute to [C,T]
        return out
    
    def denoising(self,signal):
        return wden_filter(signal)


@DATASOURCES.register_module()
class WiFi_Widar_ang_pt(BaseDataSource):
    def __init__(self, data_prefix, data_prefix_train=None, data_prefix_val=None, data_prefix_test=None,
                  multi_dataset=False,keep_antenna=True,conj_pre=False,dual=False, classes=None,prop=1.0,
                    ann_file=None, test_mode=False,color_type='color', channel_order='rgb', file_client_args=dict(backend='disk')):
        if multi_dataset:
            self.dataset = ConcatDataset([torch.load(data_prefix_train),
                           torch.load(data_prefix_val),
                           torch.load(data_prefix_test)])
        else:    
            self.dataset = torch.load(data_prefix)
        self.prop = prop
        train_len = int(prop*len(self.dataset))
        res_len = len(self.dataset)-train_len 
        self.keep_antenna = keep_antenna
        self.conj_pre = conj_pre
        self.dataset, _ = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[train_len,res_len],
            generator=torch.Generator().manual_seed(0)
        )
        super().__init__(data_prefix, classes, ann_file, test_mode, color_type, channel_order, file_client_args)

    def load_annotations(self):
        data_infos = []
        for i,(feature,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            #info['filename'] = filename
            data_infos.append(info)
        return data_infos

        
    def get_csi(self, idx):

        out = self.dataset[idx][0]
        if len(out.shape)==3:
            print(out.shape)
            raise RuntimeError
            out = torch.tensor(out,dtype=torch.float)
            out = out.reshape(out.shape[0],-1).permute(1,0)
        if len(out.shape)==4:
            if self.conj_pre == True: # do the conj mul to the csi data
                #out shape [T,C,antenna,4(amp,ang,ratio_amp，ratio_ang)]
                out = out[:,:,:,3].squeeze()  # [T,C,antenna]
                out = out[:,:,[0,2]]  # [T,C,antenna]
                if self.keep_antenna== True:
                    out = out.permute(2,1,0) # permute to [A,C,T]
                    return out
                out = out.reshape(out.shape[0],-1) #[T,C] on cpu
                out = out.permute(1,0) # permute to [C,T]
                return out
            #out shape [T,C,antenna,2(amp,phase)]
            out = out[:,:,:,1].squeeze()  # [T,C,antenna]
            if self.keep_antenna== True:
                out = out.permute(2,1,0) # permute to [A,C,T]
                return out
            out = out.reshape(out.shape[0],-1) #[T,C] on cpu
            out = out.permute(1,0) # permute to [C,T]
        return out
    
###################################################

class Intel:
    """
    class used to read csi data from .dat file
    This implementation is modified from
    https://github.com/citysu/csiread/blob/master/examples/csireadIntel5300.py
    """
    def __init__(self, file, nrxnum=3, ntxnum=2, pl_len=0, if_report=True):
        self.file = file
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        self.pl_len = pl_len    # useless
        self.if_report = if_report
        if not os.path.isfile(file):
            raise Exception("error: file does not exist, Stop!\n")

    def __getitem__(self, index):
        """Return contents of 0xbb packets"""
        ret = {
            "timestamp_low": self.timestamp_low[index],
            "bfee_count": self.bfee_count[index],
            "Nrx": self.Nrx[index],
            "Ntx": self.Ntx[index],
            "rssi_a": self.rssi_a[index],
            "rssi_b": self.rssi_b[index],
            "rssi_c": self.rssi_c[index],
            "noise": self.noise[index],
            "agc": self.agc[index],
            "perm": self.perm[index],
            "rate": self.rate[index],
            "csi": self.csi[index]
        }
        return ret

    def read(self):
        f = open(self.file, 'rb')
        if f is None:
            f.close()
            return -1

        lens = os.path.getsize(self.file)
        btype = np.int_
        #self.timestamp_low = np.zeros([lens//95], dtype = btype)
        self.timestamp_low = np.zeros([lens//95], dtype = np.int64)
        self.bfee_count = np.zeros([lens//95], dtype = btype)
        self.Nrx = np.zeros([lens//95], dtype = btype)
        self.Ntx = np.zeros([lens//95], dtype = btype)
        self.rssi_a = np.zeros([lens//95], dtype = btype)
        self.rssi_b = np.zeros([lens//95], dtype = btype)
        self.rssi_c = np.zeros([lens//95], dtype = btype)
        self.noise = np.zeros([lens//95], dtype = btype)
        self.agc = np.zeros([lens//95], dtype = btype)
        self.perm = np.zeros([lens//95, 3], dtype = btype)
        self.rate = np.zeros([lens//95], dtype = btype)
        self.csi = np.zeros([lens//95, 30, self.nrxnum, self.ntxnum], dtype = np.complex_)

        cur = 0
        count = 0
        while cur < (lens-3):
            temp = f.read(3)
            field_len = temp[1]+(temp[0]<<8)
            code = temp[2]
            cur += 3
            if code == 187:
                buf = f.read(field_len - 1)
                if len(buf) != field_len - 1:
                    break
                self.timestamp_low[count] = int.from_bytes(buf[:4], 'little')
                self.bfee_count[count] = int.from_bytes(buf[4:6], 'little')
                assert self.nrxnum == buf [8] # check the pre given nrx number is correct
                assert self.ntxnum == buf [9] # check the pre given ntx number is correct
                self.Nrx[count] = buf[8]
                self.Ntx[count] = buf[9]
                self.rssi_a[count] = buf[10]
                self.rssi_b[count] = buf[11]
                self.rssi_c[count] = buf[12]
                self.noise[count] = int.from_bytes(buf[13:14], 'little', signed=True)
                self.agc[count] = buf[14]
                self.rate[count] = int.from_bytes(buf[18:20], 'little')

                self.perm[count, 0] = buf[15] & 0x3
                self.perm[count, 1] = (buf[15] >> 2) & 0x3
                self.perm[count, 2] = (buf[15] >> 4) & 0x3

                index = 0
                payload = buf[20:]
                for i in range(30):
                    index += 3
                    remainder = index & 0x7
                    for j in range(buf[8]):
                        for k in range(buf[9]):
                            a = (payload[index // 8] >> remainder) | (payload[index // 8 + 1] << (8 - remainder)) & 0xff
                            b = (payload[index // 8 + 1] >> remainder) | (payload[index // 8 + 2] << (8 - remainder)) & 0xff
                            if a >= 128:
                                a -= 256
                            if b >= 128:
                                b -= 256
                            self.csi[count, i, self.perm[count, j], k] = a + b * 1.j
                            index += 16
                count += 1
            else:
                f.seek(field_len - 1, os.SEEK_CUR)
            cur += field_len - 1
        f.close()
        self.timestamp_low = self.timestamp_low[:count]
        self.bfee_count = self.bfee_count[:count]
        self.Nrx = self.Nrx[:count]
        self.Ntx = self.Ntx[:count]
        self.rssi_a = self.rssi_a[:count]
        self.rssi_b = self.rssi_b[:count]
        self.rssi_c = self.rssi_c[:count]
        self.noise = self.noise[:count]
        self.agc = self.agc[:count]
        self.perm = self.perm[:count, :]
        self.rate = self.rate[:count]
        self.csi = self.csi[:count, :, :, :]
        self.count = count

    def get_total_rss(self):
        """Calculates the Received Signal Strength (RSS) in dBm from CSI"""
        rssi_mag = np.zeros_like(self.rssi_a, dtype=float)
        rssi_mag += self.__dbinvs(self.rssi_a)
        rssi_mag += self.__dbinvs(self.rssi_b)
        rssi_mag += self.__dbinvs(self.rssi_c)
        ret = self.__db(rssi_mag) - 44 - self.agc
        return ret

    def get_scaled_csi(self):
        """Converts CSI to channel matrix H"""
        csi = self.csi
        csi_sq = (csi * csi.conj()).real
        csi_pwr = np.sum(csi_sq, axis=(1, 2, 3))
        rssi_pwr = self.__dbinv(self.get_total_rss())

        scale = rssi_pwr / (csi_pwr / 30)

        noise_db = self.noise
        thermal_noise_pwr = self.__dbinv(noise_db)
        thermal_noise_pwr[noise_db == -127] = self.__dbinv(-92)

        quant_error_pwr = scale * (self.Nrx * self.Ntx)
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr

        ret = self.csi * np.sqrt(scale / total_noise_pwr).reshape(-1, 1, 1, 1)
        ret[self.Ntx == 2] *= np.sqrt(2)
        ret[self.Ntx == 3] *= np.sqrt(self.__dbinv(4.5))
        ret = ret.conj()
        return ret

    def get_scaled_csi_sm(self):
        """Converts CSI to channel matrix H
        This version undoes Intel's spatial mapping to return the pure
        MIMO channel matrix H.
        """
        ret = self.get_scaled_csi()
        ret = self.__remove_sm(ret)
        return ret

    def __dbinvs(self, x):
        """Convert from decibels specially"""
        ret = np.power(10, x / 10)
        ret[ret == 1] = 0
        return ret

    def __dbinv(self, x):
        """Convert from decibels"""
        ret = np.power(10, x / 10)
        return ret

    def __db(self, x):
        """Calculates decibels"""
        ret = 10 * np.log10(x)
        return ret

    def __remove_sm(self, scaled_csi):
        """Actually undo the input spatial mapping"""
        sm_1 = 1
        sm_2_20 = np.array([[1, 1],
                            [1, -1]]) / np.sqrt(2)
        sm_2_40 = np.array([[1, 1j],
                            [1j, 1]]) / np.sqrt(2)
        sm_3_20 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 33), 2 * np.pi / (80 / 3)],
                            [ 2 * np.pi / (80 / 23), 2 * np.pi / (48 / 13), 2 * np.pi / (240 / 13)],
                            [-2 * np.pi / (80 / 13), 2 * np.pi / (240 / 37), 2 * np.pi / (48 / 13)]])
        sm_3_20 = np.exp(1j * sm_3_20) / np.sqrt(3)
        sm_3_40 = np.array([[-2 * np.pi / 16, -2 * np.pi / (80 / 13), 2 * np.pi / (80 / 23)],
                            [-2 * np.pi / (80 / 37), -2 * np.pi / (48 / 11), -2 * np.pi / (240 / 107)],
                            [ 2 * np.pi / (80 / 7), -2 * np.pi / (240 / 83), -2 * np.pi / (48 / 11)]])
        sm_3_40 = np.exp(1j * sm_3_40) / np.sqrt(3)
    
        ret = scaled_csi
        for i in range(self.count):
            M = self.Ntx[i]
            if (int(self.rate[i]) & 2048) == 2048:
                if M == 3:
                    sm = sm_3_40
                elif M == 2:
                    sm = sm_2_40
                else:
                    sm = sm_1
            else:
                if M == 3:
                    sm = sm_3_20
                elif M == 2:
                    sm = sm_2_20
                else:
                    sm = sm_1
            if sm != 1:
                ret[i, :, :, :M] = ret[i, :, :, :M].dot(sm.T.conj())
        return ret
