import os,pickle
import sys

import numpy as np
import torch 
from scipy import interpolate

import random
import logging
from tqdm import tqdm



def split(fl,train_ratio=0.8,val_ratio=0.1):
    n_total = len(fl)
    train_offset = int(n_total * train_ratio)
    val_offset = int(n_total * (train_ratio + val_ratio))
    random.shuffle(fl)
    train_list = fl[:train_offset]
    val_list = fl[train_offset:val_offset]
    test_list = fl[val_offset:]
    return train_list,val_list,test_list
def action_ratio(fl):
    act_dict = {'bed':0,'run':0,'fall':0,'walk':0,'standup':0,'pickup':0,'sitdown':0}
    for f in fl:
        for action in act_dict.keys():
            if action in f.split("_"):
                act_dict[action]+=1
    return act_dict


def falldefi_data_info(file_name):
    data_info = file_name.split('_')
    action = data_info[0]
    action = FallOrNot(action)
    room = data_info[1].split('.')[0]
    data_info = {'action':action,
                 'room':room
    }
    return data_info

def FallOrNot(ActionWithNum):
    Fall_list = ['fall','lb','lc','tr','sl']
    for fallact in Fall_list:
        if fallact in ActionWithNum[:4]:
            return 'Fall'
    return 'notFall'



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
    interval = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / new_length
    cur_range = time_stamp[0] + interval
    temp_list = []
    align_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                align_data.append(average_list(temp_list))
            else:
                align_data.append(data[i])
            temp_list = []
            cur_range = cur_range + interval
        temp_list.append(data[i])
    if len(temp_list) != 0:
        align_data.append(average_list(temp_list))
    while len(align_data) < new_length:
        align_data.append(data[len(time_stamp)-1])
        #print("shorter than new_length, add the last element")
    return align_data[:new_length]

def upsample(signal,time_stamp,newtimelen):
    '''
    根据这个位置在两个stamp之间的比例进行插值
    TODO 比较直接生成时间轴然后用np的插值函数和自己逐个插值之间时间区别
    '''
    # 需要require_new_time_stamp个元素,那么中间的间隔就需要require_new_time_stamp+1个
    require_new_time_stamp = newtimelen-len(time_stamp)
    interval = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / (require_new_time_stamp+1)
    new_time_stamp = np.zeros(newtimelen)
    new_time_stamp[0] = time_stamp[0]
    #new_time_stamp[-1] = time_stamp[-1]
    current_timestamp = time_stamp[0]
    origin_time_index = 1
    new_time_index = 1
    while new_time_index<(newtimelen):
        current_timestamp = current_timestamp+interval
        while current_timestamp>= time_stamp[origin_time_index] and origin_time_index<len(time_stamp)-1: 
            new_time_stamp[new_time_index] = time_stamp[origin_time_index]
            origin_time_index+=1
            new_time_index +=1    
        if current_timestamp >time_stamp[-1]:
            current_timestamp = (new_time_stamp[new_time_index-1]+time_stamp[-1])/2      
        new_time_stamp[new_time_index] = current_timestamp
        new_time_index+=1
    #print(signal.shape) # T C
    f = interpolate.interp1d(time_stamp, signal,axis=0)
    return f(new_time_stamp) 

def resample(signal,time_stamp,newtimelen):
    """
        upsample or downsample the signal to target length
    Args:
        signal: Original signal
        time_stamp:time_stamp of the original signal
        newtimelen: the target time length
    """
    current_time_len = len(signal)
    # upsample
    if current_time_len<newtimelen:
        return upsample(signal,time_stamp,newtimelen)

    # no action
    if current_time_len == newtimelen:
        return signal

    # downsample
    if current_time_len>newtimelen:
        return merge_timestamp(signal,time_stamp,newtimelen)


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
        annotation_list.append(int(data_info['gesture'])) # from 1 to 6
        input_list.append(f)
    assert len(annotation_list) == len(input_list)
    sample_list = zip(annotation_list,input_list)
    return sample_list

class Intel:
    """
    class used to read csi data from .dat file
    This implementation is modified from
    https://github.com/citysu/csiread/blob/master/examples/csireadIntel5300.py
    """
    def __init__(self, file, nrxnum=3, ntxnum=1, pl_len=0, if_report=True):
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

def generate_flist():
    root_path = '/media/mnt/056EB4A96239713D/xuke_arap/dataset/Falldefi/fall_detection'
    sub_list = os.listdir(root_path)
    all_file_list = []
    all_action_list = []
    all_room_list = []
    for subfolder in sub_list:
        subfolder_path = os.path.join(root_path,subfolder)
        subfolder_flist = os.listdir(subfolder_path)
        for datfile in subfolder_flist:
            fpath = os.path.join(subfolder_path,datfile)
            dinfo = falldefi_data_info(datfile)
            all_file_list.append(os.path.join(subfolder_path,datfile))
            all_action_list.append(dinfo['action'])
            all_room_list.append(dinfo['room'])
    np.savetxt('/media/mnt/056EB4A96239713D/xuke_arap/dataset/Falldefi/file_list.txt', all_file_list, delimiter=" ", fmt="%s") 
    np.savetxt('/media/mnt/056EB4A96239713D/xuke_arap/dataset/Falldefi/act_list.txt', all_action_list, delimiter=" ", fmt="%s") 
    np.savetxt('/media/mnt/056EB4A96239713D/xuke_arap/dataset/Falldefi/room_list.txt', all_room_list, delimiter=" ", fmt="%s") 

def extract_data_to_pkl(root,pklfile):
    data_file = root+pklfile
    action_list = ['Fall','notFall']
    #做数据处理，处理成为tensor dataset的格式后做数据集分割
    r_file = np.loadtxt(os.path.join(root,'file_list.txt'),dtype = str,delimiter=" ")
    r_act = np.loadtxt(os.path.join(root,'act_list.txt'),dtype = str,delimiter=" ")
    records = []
    labels = []
    timestamps = []
    for f,act in tqdm(zip(r_file,r_act)):
        csidata = Intel(f, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
        try:
            csidata.read()
        except:
            print(f)
        csi = csidata.get_scaled_csi() # shape [T,30,3,1]
        label = action_list.index(act)
        timestamp = csidata.timestamp_low
        records.append(csi)
        labels.append(label)
        timestamps.append(timestamp)
    dataset = {'record':records,'label':labels,'stamp':timestamps}
    f=open(data_file,'wb')
    pickle.dump(dataset,f)

def extract_spec_files(root,ptfile):
    action_list = ['Fall','notFall']
    #做数据处理，处理成为tensor dataset的格式后做数据集分割
    r_file = np.loadtxt(os.path.join(root,'file_list.txt'),dtype = str,delimiter=" ")
    r_act = np.loadtxt(os.path.join(root,'act_list.txt'),dtype = str,delimiter=" ")
    records = []
    labels = []
    for f,act in tqdm(zip(r_file,r_act)):
            csidata = Intel(f, nrxnum=3, ntxnum=1, pl_len=0, if_report=True)
            try:
                csidata.read()
            except:
                print(f)
            csi = csidata.get_scaled_csi() # shape [T,30,3,1]
            #复数可以表示为A*e^(jw) A是幅值，w是相位角
            amp = np.abs(csi) #shape [T,30,3,1]
            phase = np.angle(csi) #shape [T,30,3,1]
            record = np.concatenate((amp,phase),axis=-1) #shape [T,30,3,2] amp在前，phase在后
            label = action_list.index(act)
            time_stamp = csidata.timestamp_low # list length = [T]
            record = resample(record, time_stamp,10000)
            record = np.array(record)
            record = torch.tensor(record, dtype=torch.float32, requires_grad=False) # shape =  [1000,30,3,2]
            records.append(record)
            labels.append(label)

    records = torch.stack(records)
    labels = torch.tensor(labels)
    data = torch.utils.data.TensorDataset(records, labels)
    torch.save(data,os.path.join(root,ptfile))
if __name__ == "__main__":
    #generate_flist()
    root= r'D:\Shotcut\fall_data\code'
    extract_data_to_pkl(root,'falldefi.pkl')