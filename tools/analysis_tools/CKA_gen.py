# Load the basic config file
import time
import mmcv
import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import argparse

from mmselfsup.datasets import build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.apis import train_model
from mmselfsup.utils import get_root_logger
from mmselfsup.utils.collect import (dist_forward_collect,
                                     nondist_forward_collect)
from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataloader, build_dataset

from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



################## configs ######################
seed = 47
sup_feature_path = ''
################################################

def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('configfile', help='train config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        help='(Deprecated, please use --work-dir) the dir to save logs and '
        'models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--dataset_config',
        default='configs/benchmarks/classification/tsne_imagenet.py',
        help='(Deprecated, please use --dataset-config) '
        'extract dataset config file path')
    parser.add_argument(
        '--max_num_class',
        type=int,
        default=22,
        help='(Deprecated, please use --max-num-class) the maximum number '
        'of classes to apply t-SNE algorithms, now the function supports '
        'maximum 20 classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args



class ExtractProcess(object):
    """feature extraction process.

    This process extracts feature map of backbone

    Args:

    """

    def __init__(self):
        pass
    def _forward_func(self, model, **x):
        """The forward function of extract process."""
        backbone_feats = model(mode='extract', **x)  #tuple(tensor) batch 320
        flat_feat = backbone_feats[0]
        return dict(feat=flat_feat.cpu())

    def extract(self, model, data_loader, distributed=False):
        """The extract function to apply forward function and choose
        distributed or not."""
        model.eval()

        # the function sent to collect function
        def func(**x):
            return self._forward_func(model, **x)

        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, data_loader, rank,
                                           len(data_loader.dataset))

        else:
            results = nondist_forward_collect(func, data_loader,
                                              len(data_loader.dataset))#445
        return results


def main():
    #os.environ['RANK']='0' 
    args = parse_args()
    cfg = Config.fromfile(args.configfile)
    cfg.work_dir = args.work_dir

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir and init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cka_work_dir = osp.join(cfg.work_dir, f'cka_{timestamp}/')
    mmcv.mkdir_or_exist(osp.abspath(cka_work_dir))
    log_file = osp.join(cka_work_dir, 'extract.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)


    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset = build_dataset(dataset_cfg.data.extract)

    # compress dataset, select that the label is less then max_num_class # TODO  这里在做什么
    tmp_infos = []
    for i in range(len(dataset)):
        if dataset.data_source.data_infos[i]['gt_label'] < args.max_num_class:
            tmp_infos.append(dataset.data_source.data_infos[i])
    dataset.data_source.data_infos = tmp_infos
    logger.info(f'Apply t-SNE to visualize {len(dataset)} samples.')

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated. '
                        'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=dataset_cfg.data.samples_per_gpu,
        workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model
    # tune the model
    PATCH_SIZE = (10,10) # 10 for widar; try other for csi office # int or tuple(H,W)
    # cfg.model = dict(
    # type='Classification_CSI',
    # img_like_tensor=True,
    # backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     in_channels=3,
    #     out_indices=[4],  # 0: conv-1, x: stage-x
    #     norm_cfg=dict(type='BN')),
    # head=dict(type='ClsCsiHead', with_avg_pool=True, in_channels=512,
    #     num_classes=22), # widar all 22; widar user123 6; OR 7; Signfi 276
    # AE_model = False
    # )
    cfg.model = dict(
    type='Classification_CSI_VIT',
    backbone=dict(
        type='VisionTransformer_CSI',
        arch='csi-small',  # embed_dim = 384 # for widar 'b' is good enough; for office room we try csi-small
        img_size=(60,500), # (30,500) for widar;(30,2000) for csi office; (30,200) for signfi; (60,500) for widar_all
        patch_size=(10,10),  # 10 for widar; try other for csi office # int or tuple(H,W)
        stop_grad_conv1=True),
    head=dict(
        type='ClsCsiHead',
        in_channels=768,
        num_classes=22, # widar all 22; widar user123 6; OR 7; Signfi 276
        vit_backbone=True,
    ))

    model = build_algorithm(cfg.model)
    model.init_weights()


    # model is determined in this priority: init_cfg > checkpoint > random
    if hasattr(cfg.model.backbone, 'init_cfg'): 
        if getattr(cfg.model.backbone.init_cfg, 'type', None) == 'Pretrained':
            logger.info(
                f'Use pretrained model: '
                f'{cfg.model.backbone.init_cfg.checkpoint} to extract features'
            )
    elif args.checkpoint is not None: 
        logger.info(f'Use checkpoint: {args.checkpoint} to extract features')
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        logger.info('No pretrained or checkpoint is given, use random init.')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)


    # build extraction processor and run
    extractor = ExtractProcess()
    features = extractor.extract(model, data_loader, distributed=distributed)
    labels = dataset.data_source.get_gt_labels()

    # save features
    mmcv.mkdir_or_exist(f'{cka_work_dir}features/')
    logger.info(f'Save features to {cka_work_dir}features/')
    if distributed:
        rank, _ = get_dist_info()
        if rank == 0:
            for key, val in features.items():
                output_file = \
                    f'{cka_work_dir}features/{dataset_cfg.name}_{key}.npy'
                np.save(output_file, val)
    else:
        for key, val in features.items():
            output_file = \
                f'{cka_work_dir}features/{dataset_cfg.name}_{key}.npy'
            np.save(output_file, val)



if __name__ == '__main__':
    main()

# X = np.load('/work_dir/ssl/simclr_causaulnet10_1xb64_Perm3_coslr-200e_csioffice/CKA/cka_20220615_183955/features/csioffice_val_feat.npy')
# Y = np.load('/work_dir/sup_baseline/sup_causualnet_1xb64-coslr-100e/CKA/cka_20220609_145527/features/csioffice_val_feat.npy')
# CKA(X,Y)

