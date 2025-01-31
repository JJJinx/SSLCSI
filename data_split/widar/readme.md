
*f_list.txt files denotes the contents of train, validation and test datasets.
.py file is used for generating the .pt dataset.

**Step 1** Download data from https://tns.thss.tsinghua.edu.cn/widar3.0/

**Step 2** Unzip and modify the folder as following form:

widar/

├── 20240201_1/

│   ├── user1/

│   │   └── sample_raw.dat

│   ├── user2/

│   │   └── sample_raw.dat

│   ├── user3/

│       └── sample_raw.dat

├── 20240201_2/

│   ├── user1/

│   │   └── sample_raw.dat

│   ├── user2/

│   │   └── sample_raw.dat

│   ├── user3/

│       └── sample_raw.dat

Generated dataset contains **record**  = (amp,phase,csi_ratio_amp,csi_ratio_ang) with a shape of  [T,30,3,4] and **label**.
