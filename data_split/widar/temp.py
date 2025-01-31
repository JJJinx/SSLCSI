import torch
from torch.utils.data import ConcatDataset

class CSIdataset(torch.utils.data.Dataset):
    def __init__(self,record,label,files):
        self.record = record
        #self.ratio_record = ratio_record
        self.label = label
        self.files = files
    def __getitem__(self,idx):
        record = self.record[idx]
        #ratio_record = self.ratio_record[idx]
        label=self.label[idx]
        filename = self.files[idx]
        return record,label,filename

    def __len__(self):
        return len(self.record)
#generate a dataset that have (amp,ratio_ang,label)
dataset = torch.load("/home/x/xuk16/data/widar_all_r2_conjugate/widar_r2_train.pt")
for record,label in dataset:
    print(record.shape)
    raise RuntimeError
