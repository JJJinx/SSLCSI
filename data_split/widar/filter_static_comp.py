import torch
from scipy.signal import butter, lfilter, freqz,sosfilt
import numpy as np

train_data = torch.load("/home/x/xuk16/data/widar_all_r2_conjugate/widar_r2_train.pt")
for feature,label in train_data: 
    print(feature.shape,label)

    raise RuntimeError
#用于处理本质上已经被conjugate multiplication 处理过后，消去了主要相位噪音后的数据
#在那样数据的基础上滤除静态路径分量
train_data = torch.load("/home/x/xuk16/data/widar_all_r2/widar_r2_train.pt")
val_data = torch.load("/home/x/xuk16/data/widar_all_r2/widar_r2_val.pt")
test_data = torch.load("/home/x/xuk16/data/widar_all_r2/widar_r2_test.pt")
# feature shape [500, 30, 3, 2(amp,phase)]
# 使用butterworth滤波器
lb,la = butter(6, 60, 'lowpass', fs=1000, output='ba') #保留低于60 HZ部分
hb,ha = butter(3, 2, 'highpass', fs=1000, output='ba') #保留高于2 HZ部分

# TODO filtered = lfilter(lu,ld, conjugated_data,axis=0)
#filtered = lfilter(lb,la, sig,axis=0)
#filtered = lfilter(hb,ha , filtered,axis=0)
# get the conjugated csi data

features = []
labels = []
for feature,label in train_data: 
    #feauture shape = [500,30,2,3(amp,conj_amp,conj_ang)]
    amp = np.expand_dims(feature[:,:,:,0],axis=-1)
    complex_feature = feature[:,:,:,1].squeeze()*torch.exp(1j*feature[:,:,:,2].squeeze())
    filtered = lfilter(lb,la, complex_feature,axis=0)
    filtered = lfilter(hb,ha , filtered,axis=0)
    conj_amp = np.expand_dims(np.abs(filtered),axis=-1)
    conj_ang = np.expand_dims(np.angle(filtered),axis=-1)
    record = np.concatenate((amp,conj_amp,conj_ang),axis=-1)
    record = torch.tensor(record, dtype=torch.float32, requires_grad=False)
    features.append(record)
    labels.append(label)

features = torch.stack(features)
labels = torch.tensor(labels)
train_set = torch.utils.data.TensorDataset(features, labels)
torch.save(train_set,'/home/x/xuk16/data/widar_all_r2_conjugate/widar_r2_train_filtered.pt')

features = []
labels = []
for feature,label in val_data: 
    #feauture shape = [500,30,2,3(amp,conj_amp,conj_ang)]
    amp = np.expand_dims(feature[:,:,:,0],axis=-1)
    complex_feature = feature[:,:,:,1].squeeze()*torch.exp(1j*feature[:,:,:,2].squeeze())
    filtered = lfilter(lb,la, complex_feature,axis=0)
    filtered = lfilter(hb,ha , filtered,axis=0)
    conj_amp = np.expand_dims(np.abs(filtered),axis=-1)
    conj_ang = np.expand_dims(np.angle(filtered),axis=-1)
    record = np.concatenate((amp,conj_amp,conj_ang),axis=-1)
    record = torch.tensor(record, dtype=torch.float32, requires_grad=False)
    features.append(record)
    labels.append(label)

features = torch.stack(features)
labels = torch.tensor(labels)
val_set = torch.utils.data.TensorDataset(features, labels)
torch.save(val_set,'/home/x/xuk16/data/widar_all_r2_conjugate/widar_r2_val_filtered.pt')

features = []
labels = []
for feature,label in test_data: 
    #feauture shape = [500,30,2,3(amp,conj_amp,conj_ang)]
    amp = np.expand_dims(feature[:,:,:,0],axis=-1)
    complex_feature = feature[:,:,:,1].squeeze()*torch.exp(1j*feature[:,:,:,2].squeeze())
    filtered = lfilter(lb,la, complex_feature,axis=0)
    filtered = lfilter(hb,ha , filtered,axis=0)
    conj_amp = np.expand_dims(np.abs(filtered),axis=-1)
    conj_ang = np.expand_dims(np.angle(filtered),axis=-1)
    record = np.concatenate((amp,conj_amp,conj_ang),axis=-1)
    record = torch.tensor(record, dtype=torch.float32, requires_grad=False)
    features.append(record)
    labels.append(label)

features = torch.stack(features)
labels = torch.tensor(labels)
test_set = torch.utils.data.TensorDataset(features, labels)
torch.save(test_set,'/home/x/xuk16/data/widar_all_r2_conjugate/widar_r2_test_filtered.pt')
