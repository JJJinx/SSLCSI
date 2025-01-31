import numpy as np
import torch
from torch.utils.data import ConcatDataset


self.dataset = ConcatDataset([torch.load(data_prefix_train),
                           torch.load(data_prefix_val),
                           torch.load(data_prefix_test)])
dataset = torch.load('/home/x/xuk16/SSLBM/widar_all_r2/widar_r2.pt')
train_size = int(0.6*len(dataset))
val_size = int(0.2*len(dataset))
test_size = len(dataset)-train_size-val_size
train_set,val_set,test_set=torch.utils.data.random_split(dataset,[train_size,val_size,test_size], generator=torch.Generator().manual_seed(42))


