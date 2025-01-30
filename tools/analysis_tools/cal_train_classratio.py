# import argparse
# from multiprocessing.pool import RUN
# import os.path as osp
# from pdb import runeval
# import time
# from typing_extensions import runtime

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import mmcv
# import numpy as np
# import torch
# from mmcv import Config, DictAction
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from sklearn.manifold import TSNE

# from mmselfsup.apis import set_random_seed
# from mmselfsup.datasets import build_dataloader, build_dataset
# from mmselfsup.models import build_algorithm
# from mmselfsup.models.utils import MultiExtractProcess,ExtractProcess
# from mmselfsup.utils import get_root_logger


# def parse_args():
#     parser = argparse.ArgumentParser(description='t-SNE visualization')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--checkpoint', default=None, help='checkpoint file')
#     parser.add_argument(
#         '--work_dir',
#         help='(Deprecated, please use --work-dir) the dir to save logs and '
#         'models')
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument(
#         '--dataset_config',
#         default='configs/benchmarks/classification/tsne_imagenet.py',
#         help='(Deprecated, please use --dataset-config) '
#         'extract dataset config file path')
#     parser.add_argument(
#         '--dataset-config',
#         default='configs/benchmarks/classification/tsne_imagenet.py',
#         help='extract dataset config file path')
#     parser.add_argument(
#         '--layer_ind',
#         type=str,
#         default='0,1,2,3,4',
#         help='(Deprecated, please use --layer-ind) layer indices, '
#         'separated by comma, e.g., "0,1,2,3,4"')
#     parser.add_argument(
#         '--layer-ind',
#         type=str,
#         default='0,1,2,3,4',
#         help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
#     parser.add_argument(
#         '--pool_type',
#         choices=['specified', 'adaptive'],
#         default='specified',
#         help='(Deprecated, please use --pool-type) Pooling type in '
#         ':class:`MultiPooling`')
#     parser.add_argument(
#         '--pool-type',
#         choices=['specified', 'adaptive'],
#         default='specified',
#         help='Pooling type in :class:`MultiPooling`')
#     parser.add_argument(
#         '--max_num_class',
#         type=int,
#         default=22,
#         help='(Deprecated, please use --max-num-class) the maximum number '
#         'of classes to apply t-SNE algorithms, now the function supports '
#         'maximum 20 classes')
#     parser.add_argument(
#         '--max-num-class',
#         type=int,
#         default=20,
#         help='the maximum number of classes to apply t-SNE algorithms, now the'
#         'function supports maximum 20 classes')
#     parser.add_argument('--seed', type=int, default=0, help='random seed')
#     parser.add_argument(
#         '--deterministic',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')

#     # t-SNE settings
#     parser.add_argument(
#         '--n_components',
#         type=int,
#         default=2,
#         help='(Deprecated, please use --n-components) the dimension of results'
#     )
#     parser.add_argument(
#         '--perplexity',
#         type=float,
#         default=30.0,
#         help='The perplexity is related to the number of nearest neighbors'
#         'that is used in other manifold learning algorithms. Should smaller than point per cluster.')
#     parser.add_argument(
#         '--early_exaggeration',
#         type=float,
#         default=12.0,
#         help='(Deprecated, please use --early-exaggeration) Controls how '
#         'tight natural clusters in the original space are in the embedded '
#         'space and how much space will be between them.')
#     parser.add_argument(
#         '--learning_rate',
#         type=float,
#         default=200.0,
#         help='(Deprecated, please use --learning-rate) The learning rate '
#         'for t-SNE is usually in the range [10.0, 1000.0]. '
#         'If the learning rate is too high, the data may look'
#         'like a ball with any point approximately equidistant from its nearest'
#         'neighbours. If the learning rate is too low, most points may look'
#         'compressed in a dense cloud with few outliers.')
#     parser.add_argument(
#         '--n_iter',
#         type=int,
#         default=1000,
#         help='(Deprecated, please use --n-iter) Maximum number of iterations '
#         'for the optimization. Should be at least 250.')
#     parser.add_argument(
#         '--n_iter_without_progress',
#         type=int,
#         default=300,
#         help='(Deprecated, please use --n-iter-without-progress) Maximum '
#         'number of iterations without progress before we abort the '
#         'optimization.')
#     parser.add_argument(
#         '--init', type=str, default='random', help='The init method')
#     parser.add_argument('--tsne_seed', type=int, default=0, help='random seed')
#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_args()

#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#     # set cudnn_benchmark
#     if cfg.get('cudnn_benchmark', False):
#         torch.backends.cudnn.benchmark = True
#     # work_dir is determined in this priority: CLI > segment in file > filename
#     if args.work_dir is not None:
#         # update configs according to CLI args if args.work_dir is not None
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         # use config filename as default work_dir if cfg.work_dir is None
#         work_type = args.config.split('/')[1]
#         cfg.work_dir = osp.join('./work_dirs', work_type,
#                                 osp.splitext(osp.basename(args.config))[0])

#     # get out_indices from args TODO out indice不知道干啥的
#     # layer_ind = [int(idx) for idx in args.layer_ind.split(',')]
#     # cfg.model.backbone.out_indices = layer_ind

#     # init distributed env first, since logger depends on the dist info.
#     if args.launcher == 'none':
#         distributed = False
#     else:
#         distributed = True
#         init_dist(args.launcher, **cfg.dist_params)

#     # create work_dir and init the logger before other steps
#     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     tsne_work_dir = osp.join(cfg.work_dir, f'umap_{timestamp}/')#osp.join(cfg.work_dir, f'tsne_{timestamp}/')
#     mmcv.mkdir_or_exist(osp.abspath(tsne_work_dir))
#     log_file = osp.join(tsne_work_dir, 'extract.log')
#     logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

#     # set random seeds
#     if args.seed is not None:
#         logger.info(f'Set random seed to {args.seed}, '
#                     f'deterministic: {args.deterministic}')
#         set_random_seed(args.seed, deterministic=args.deterministic)

#     # build the dataloader
#     #dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
#     #dataset = build_dataset(dataset_cfg.data.extract)
#     dataset = build_dataset(cfg.data.extract)
#     # compress dataset, select that the label is less then max_num_class
#     tmp_infos = []
#     for i in range(len(dataset)):
#         if dataset.data_source.data_infos[i]['gt_label'] < args.max_num_class:
#             tmp_infos.append(dataset.data_source.data_infos[i])
#     dataset.data_source.data_infos = tmp_infos
#     labels = dataset.data_source.get_gt_labels()
#     logger.info(f'labels {labels}')

# if __name__ == '__main__':
#     main()


import torch
import numpy as np

home_val = np.load('/data/huaweipt/val_home.npy',allow_pickle=True)
office_val = np.load('/data/huaweipt/val_office.npy',allow_pickle=True)
home_val = home_val.item()
home_val_labels = home_val['label'] #shape (N,) <class 'numpy.ndarray'>
print(home_val_labels.shape)
office_val = office_val.item()
office_val_labels = office_val['label'] #shape (N,) <class 'numpy.ndarray'>
print(office_val_labels.shape)

raise RuntimeError
# dataset
train_a_list = [        
        '/data/huawei_data_100_sl/20230911/room0',
        '/data/huawei_data_100_sl/20230911/room1',
        '/data/huawei_data_100_sl/20230911/room3',
        '/data/huawei_data_100_sl/20230918/room0',
        '/data/huawei_data_100_sl/20230918/room1',
        '/data/huawei_data_100_sl/20230918/room2',
        '/data/huawei_data_100_sl/20230918/room3',
        '/data/huawei_data_100_sl/20230925/room0',
        '/data/huawei_data_100_sl/20230925/room1',
        '/data/huawei_data_100_sl/20230925/room2',
        '/data/huawei_data_100_sl/20230925/room3',
        '/data/huawei_data_100_sl/20231016/room0_door1',
        '/data/huawei_data_100_sl/20231016/room0_door2',
        '/data/huawei_data_100_sl/20231016/room1_door1',
        '/data/huawei_data_100_sl/20231016/room1_door2',
        '/data/huawei_data_100_sl/20231016/room2_door1',
        '/data/huawei_data_100_sl/20231016/room2_door2',
        '/data/huawei_data_100_sl/20231016/room3_door1',
        '/data/huawei_data_100_sl/20231016/room3_door2',
        '/data/huawei_data_100_sl/20231020/AP_m',
        '/data/huawei_data_100_sl/20231020/AP_r',
        '/data/huawei_data_100_sl/20231020/AP_l',
        '/data/huawei_data_100_sl/20231030/scene1_roomA',
        '/data/huawei_data_100_sl/20231030/scene1_roomB',
        '/data/huawei_data_100_sl/20231030/scene2_roomA',
        '/data/huawei_data_100_sl/20231030/scene2_roomB',
        ]
val_a_list = ['/data/huawei_data/20231016/room2_door1',
            '/data/huawei_data/20231016/room2_door2',
            '/data/huawei_data/20231020/AP_m',
            ]
test_a_list = ['/data/huawei_data/20231016/room3_door1',
               '/data/huawei_data/20231016/room3_door2',
               '/data/huawei_data/20231020/AP_r',
               ]





train_dataset = WiFi_Huawei_ang_sl(train_a_list,val_a_list,test_a_list,mode='train',data_prefix=None)

dataset_len = len(train_dataset)

labels = train_dataset.load_annotations()
print('total samples=',dataset_len)

positive_samples = 0
negtive_samples = 0
non = 0
for item in labels:
    if item['living_label']>0:
        positive_samples+=1
    if item['living_label']==0:
        negtive_samples+=1
    else:
        non+=1
print('positive_samples={},negtive_samples={}'.format(positive_samples,negtive_samples))    