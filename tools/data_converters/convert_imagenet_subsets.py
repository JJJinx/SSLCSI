# Copyright (c) OpenMMLab. All rights reserved.
"""SimCLR provides list files for semi-supervised benchmarks:

https://github.com/google-research/simclr/tree/master/imagenet_subsets/
This script converts the list files into the required format in MMSelfSup.
"""
import argparse

parser = argparse.ArgumentParser(
    description='Convert ImageNet subset lists provided by SimCLR.')
parser.add_argument('input', help='Input list file.')
parser.add_argument('output', help='Output list file.')
args = parser.parse_args()

# create dict
with open('data/imagenet/meta/train_labeled.txt', 'r') as f:
    lines = f.readlines()
keys = [line.split('/')[0] for line in lines]
labels = [line.strip().split()[1] for line in lines]
mapping = {}
for k, l in zip(keys, labels):
    if k not in mapping:
        mapping[k] = l
    else:
        assert mapping[k] == l

# convert
with open(args.input, 'r') as f:
    lines = f.readlines()
fns = [line.strip() for line in lines]
sample_keys = [line.split('_')[0] for line in lines]
sample_labels = [mapping[k] for k in sample_keys]
output_lines = [
    f'{k}/{fn} {l}\n' for k, fn, l in zip(sample_keys, fns, sample_labels)
]
with open(args.output, 'w') as f:
    f.writelines(output_lines)
