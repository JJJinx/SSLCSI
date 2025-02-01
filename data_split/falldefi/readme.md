Download data from https://github.com/dmsp123/FallDeFi.  
During the use of falldefi, we discarded the following data:
1. **Kitchen data**: The action labels used unclear abbreviations.
2. **LoS data**: Some entries had a different tx number; most of the data in this folder had `txnumber=2`, while the rest had `txnumber=1`.

After running `generate.py`, the generated `.pkl` file is a dictionary with the following components:
- **record**: A list of length 553, where each entry is a complex tensor with shape (9792, 30, 3, 1).
- **label**: A list of length 553, where each entry is either 0 (not a fall) or 1 (fall).
- **stamp**: A list of length 553, where each element is a list of timestamps.

