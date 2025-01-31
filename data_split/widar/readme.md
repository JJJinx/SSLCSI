This .py file is used to generate the .pt dataset.

**Step 1** Download data from https://tns.thss.tsinghua.edu.cn/widar3.0/

**Step 2** Unzip and modify the folder as following form:
widar/  
├── 20181108/  
│   ├── user1/  
│   │   └── sample_raw.dat  
│   ├── user2/  
│   │   └── sample_raw.dat  
│   ├── user3/   
│       └── sample_raw.dat  
├── 20181117/  
│   ├── user3/   
│   │   └── sample_raw.dat   
│   ├── user7/   
│   │   └── sample_raw.dat   
│   ├── user8/   
│       └── sample_raw.dat   

**Step 3** Run generate_split.py

The generated dataset contains:
> Record: (amp, phase, csi_ratio_amp, csi_ratio_ang)  with a shape: [T, 30, 3, 4]  
> Label: Corresponding label information.  

Additionally, the following files are created to define the dataset split for reproducibility:
> train_f_list.txt  
> val_f_list.txt  
> test_f_list.txt  
These files contain the respective data splits for training, validation, and testing.
