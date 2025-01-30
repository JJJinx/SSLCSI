from multiprocessing.pool import RUN
import re,os,pickle
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
class CSIDA(BaseDataSource):
    def __init__(self, data_prefix,all_data,split_seed,mode,conj_pre,dual,prop=1.0,multi_dataset=False,keep_antenna=True):
        """
        Args:
            data_prefix (str): the root path of CSIDA in .pkl form.
            all_data (dict): a dict in the form of {'roomid':[list],'locid':[list],'userid':[list]}
                          lists in the same dict should have the same length
            test (dict): a dict in the form of {'roomid':[list],'locid':[list],'userid':[list]} 
                          lists in the smae dict should have the same length
        Returns:
            None
        """
        
        self.conj_pre = conj_pre
        self.dual = dual
        self.keep_antenna = keep_antenna
        all_sel_amp = []
        all_sel_pha = []
        all_sel_gesture = []
        for roomid, locid, userid in zip(all_data['roomid'], all_data['locid'], all_data['userid']):
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl','rb')
            print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl')
            sel_amp=pickle.load(f)    # np.array (N,A,C,T) (113, 3, 114, 1800)
            f.close()
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl','rb')
            print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl')
            sel_pha=pickle.load(f)  # np.array (N,A,C,T) (113, 3, 114, 1800)
            f.close()
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_label.pkl','rb')
            sel_gesture = pickle.load(f) # np.array (N,) (113,)
            f.close()  
            all_sel_amp.append(sel_amp)
            all_sel_pha.append(sel_pha)
            all_sel_gesture.append(sel_gesture)
        all_sel_amp = torch.from_numpy(np.concatenate(all_sel_amp))
        all_sel_pha = torch.from_numpy(np.concatenate(all_sel_pha))
        all_sel_gesture = torch.from_numpy(np.concatenate(all_sel_gesture))
        # calculate conj-phase
        real = all_sel_amp*torch.cos(all_sel_pha) 
        img = all_sel_amp*torch.sin(all_sel_pha) 
        compx = torch.complex(real,img)
        ref_ant = torch.unsqueeze(torch.conj(compx[:,1,:,:]),axis=1) # shape [N,1,C,T]
        all_conj_pha = torch.angle(compx*ref_ant)[:,[0,2],:,:]
        dataset = torch.utils.data.TensorDataset(all_sel_amp,all_sel_pha,all_conj_pha,all_sel_gesture)

        proportions = [.8, .2]
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        train_dataset,test_dataset = torch.utils.data.random_split(
            dataset,lengths,generator=torch.Generator().manual_seed(split_seed))
        
        proportions = [.9, .1]
        lengths = [int(p * len(train_dataset)) for p in proportions]
        lengths[-1] = len(train_dataset) - sum(lengths[:-1])
        train_dataset,val_dataset = torch.utils.data.random_split(
            train_dataset,lengths)
        if mode == 'train':
            self.dataset = train_dataset
        if mode == 'val':
            self.dataset = val_dataset
        if mode == 'test':
            self.dataset = test_dataset
        super().__init__(data_prefix)

    def load_annotations(self):
        data_infos = []
        for i,(amp,pha,conj_pha,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        """Get CSI sample by index.

        Args:
            idx (int): Index of data. 
        Returns:
            record: CSI signal shape = [C T] or [A C T].
        """
        amp,pha,conj_pha,_ = self.dataset[idx]
        if self.dual == 'amp':
            out = amp  # shape [A,C,T] [3,114,1800]
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        if self.dual == 'pha':
            out = pha  # shape [A,C,T]
            if self.conj_pre == True:
                out = conj_pha   #[2, 114, 1800]                
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        if self.dual == 'sep': # shape should be [A,C,T,2]
            if self.conj_pre == True:
                amp = amp[[0,2],:,:]
                out = torch.stack((amp,conj_pha),dim = -1)
            else:
                out = torch.stack((amp,pha),dim = -1)
            if self.keep_antenna == False: # output shape should be [AC,T,2]
                time_length = out.shape[2]
                out = out.permute(2,3,0,1).reshape(time_length,2,-1).permute(2,0,1) #shape [AC,T,2]
        if self.dual == 'con':   # shape should be [A,2C,T]
            if self.conj_pre == True:
                amp = amp[[0,2],:,:]
                out = torch.cat((amp,conj_pha),dim=1)
            else:
                out = torch.cat((amp,pha),dim=1)
            if self.keep_antenna == False: # output shape should be [2AC,T]
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [2AC,T]
        return out


@DATASOURCES.register_module()
class CSIDA_CrossDomain(BaseDataSource):
    def __init__(self, data_prefix,all_data,split_seed,mode,conj_pre,dual,prop=1.0,multi_dataset=False,keep_antenna=True,
                selected_room=-1,selected_loc=-1,selected_user=-1):
        """
        Args:
            data_prefix (str): the root path of CSIDA in .pkl form.
            all_data (dict): a dict in the form of {'roomid':[list],'locid':[list],'userid':[list]}
                          lists in the same dict should have the same length
            test (dict): a dict in the form of {'roomid':[list],'locid':[list],'userid':[list]} 
                          lists in the smae dict should have the same length
        Returns:
            None

        """
        #'roomid': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,],
        #'locid' : [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,0,0,0,0,0,1,1,1,1,1,],
        #'userid': [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,],
        self.conj_pre = conj_pre
        self.dual = dual
        self.keep_antenna = keep_antenna
        all_sel_amp = []
        all_sel_pha = []
        all_sel_gesture = []
        for roomid, locid, userid in zip(all_data['roomid'], all_data['locid'], all_data['userid']):
            if selected_room>=0:
                roomid = selected_room
            if selected_loc>=0:
                locid = selected_loc
            if selected_user>=0:
                userid = selected_user
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl','rb')
            print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_amp.pkl')
            sel_amp=pickle.load(f)    # np.array (N,A,C,T) (113, 3, 114, 1800)
            f.close()
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl','rb')
            print('extracting:',data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_pha.pkl')
            sel_pha=pickle.load(f)  # np.array (N,A,C,T) (113, 3, 114, 1800)
            f.close()
            f=open(data_prefix+'room_'+str(roomid)+'_loc_'+str(locid)+'_user_'+str(userid)+'_CSIDA_label.pkl','rb')
            sel_gesture = pickle.load(f) # np.array (N,) (113,)
            f.close()  
            all_sel_amp.append(sel_amp)
            all_sel_pha.append(sel_pha)
            all_sel_gesture.append(sel_gesture)
        all_sel_amp = torch.from_numpy(np.concatenate(all_sel_amp))
        all_sel_pha = torch.from_numpy(np.concatenate(all_sel_pha))
        all_sel_gesture = torch.from_numpy(np.concatenate(all_sel_gesture))
        # calculate conj-phase
        real = all_sel_amp*torch.cos(all_sel_pha) 
        img = all_sel_amp*torch.sin(all_sel_pha) 
        compx = torch.complex(real,img)
        ref_ant = torch.unsqueeze(torch.conj(compx[:,1,:,:]),axis=1) # shape [N,1,C,T]
        all_conj_pha = torch.angle(compx*ref_ant)[:,[0,2],:,:]
        dataset = torch.utils.data.TensorDataset(all_sel_amp,all_sel_pha,all_conj_pha,all_sel_gesture)

        proportions = [.8, .2]
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        train_dataset,test_dataset = torch.utils.data.random_split(
            dataset,lengths,generator=torch.Generator().manual_seed(split_seed))
        
        proportions = [.9, .1]
        lengths = [int(p * len(train_dataset)) for p in proportions]
        lengths[-1] = len(train_dataset) - sum(lengths[:-1])
        train_dataset,val_dataset = torch.utils.data.random_split(
            train_dataset,lengths)
        if mode == 'all':
            self.dataset = dataset
        if mode == 'train':
            self.dataset = train_dataset
        if mode == 'val':
            self.dataset = val_dataset
        if mode == 'test':
            self.dataset = test_dataset
        super().__init__(data_prefix)

    def load_annotations(self):
        data_infos = []
        for i,(amp,pha,conj_pha,gt_label) in enumerate(self.dataset):
            info = {}
            info['gt_label'] = torch.tensor(gt_label,dtype = torch.long)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
    
    def get_csi(self, idx):
        """Get CSI sample by index.

        Args:
            idx (int): Index of data. 
        Returns:
            record: CSI signal shape = [C T] or [A C T].
        """
        amp,pha,conj_pha,_ = self.dataset[idx]
        if self.dual == 'amp':
            out = amp  # shape [A,C,T] [3,114,1800]
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        if self.dual == 'pha':
            out = pha  # shape [A,C,T]
            if self.conj_pre == True:
                out = conj_pha   #[2, 114, 1800]                
            if self.keep_antenna == False:
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [AC,T]
        if self.dual == 'sep': # shape should be [A,C,T,2]
            if self.conj_pre == True:
                amp = amp[[0,2],:,:]
                out = torch.stack((amp,conj_pha),dim = -1)
            else:
                out = torch.stack((amp,pha),dim = -1)
            if self.keep_antenna == False: # output shape should be [AC,T,2]
                time_length = out.shape[2]
                out = out.permute(2,3,0,1).reshape(time_length,2,-1).permute(2,0,1) #shape [AC,T,2]
        if self.dual == 'con':   # shape should be [A,2C,T]
            if self.conj_pre == True:
                amp = amp[[0,2],:,:]
                out = torch.cat((amp,conj_pha),dim=1)
            else:
                out = torch.cat((amp,pha),dim=1)
            if self.keep_antenna == False: # output shape should be [2AC,T]
                time_length = out.shape[2]
                out = out.permute(2,0,1).reshape(time_length,-1).permute(1,0) #shape [2AC,T]
        return out