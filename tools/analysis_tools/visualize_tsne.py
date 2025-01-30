# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from multiprocessing.pool import RUN
import os.path as osp
from pdb import runeval
import time
from typing_extensions import runtime

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from sklearn.manifold import TSNE
import umap

from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import MultiExtractProcess,ExtractProcess
from mmselfsup.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        help='(Deprecated, please use --work-dir) the dir to save logs and '
        'models')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--dataset_config',
        default='configs/benchmarks/classification/tsne_imagenet.py',
        help='(Deprecated, please use --dataset-config) '
        'extract dataset config file path')
    parser.add_argument(
        '--dataset-config',
        default='configs/benchmarks/classification/tsne_imagenet.py',
        help='extract dataset config file path')
    parser.add_argument(
        '--layer_ind',
        type=str,
        default='0,1,2,3,4',
        help='(Deprecated, please use --layer-ind) layer indices, '
        'separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--layer-ind',
        type=str,
        default='0,1,2,3,4',
        help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--pool_type',
        choices=['specified', 'adaptive'],
        default='specified',
        help='(Deprecated, please use --pool-type) Pooling type in '
        ':class:`MultiPooling`')
    parser.add_argument(
        '--pool-type',
        choices=['specified', 'adaptive'],
        default='specified',
        help='Pooling type in :class:`MultiPooling`')
    parser.add_argument(
        '--max_num_class',
        type=int,
        default=22,
        help='(Deprecated, please use --max-num-class) the maximum number '
        'of classes to apply t-SNE algorithms, now the function supports '
        'maximum 20 classes')
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=20,
        help='the maximum number of classes to apply t-SNE algorithms, now the'
        'function supports maximum 20 classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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

    # t-SNE settings
    parser.add_argument(
        '--n_components',
        type=int,
        default=2,
        help='(Deprecated, please use --n-components) the dimension of results'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms. Should smaller than point per cluster.')
    parser.add_argument(
        '--early_exaggeration',
        type=float,
        default=12.0,
        help='(Deprecated, please use --early-exaggeration) Controls how '
        'tight natural clusters in the original space are in the embedded '
        'space and how much space will be between them.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=200.0,
        help='(Deprecated, please use --learning-rate) The learning rate '
        'for t-SNE is usually in the range [10.0, 1000.0]. '
        'If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='(Deprecated, please use --n-iter) Maximum number of iterations '
        'for the optimization. Should be at least 250.')
    parser.add_argument(
        '--n_iter_without_progress',
        type=int,
        default=300,
        help='(Deprecated, please use --n-iter-without-progress) Maximum '
        'number of iterations without progress before we abort the '
        'optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')
    parser.add_argument('--tsne_seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # get out_indices from args TODO out indice不知道干啥的
    # layer_ind = [int(idx) for idx in args.layer_ind.split(',')]
    # cfg.model.backbone.out_indices = layer_ind

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir and init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    tsne_work_dir = osp.join(cfg.work_dir, f'umap_{timestamp}/')#osp.join(cfg.work_dir, f'tsne_{timestamp}/')
    mmcv.mkdir_or_exist(osp.abspath(tsne_work_dir))
    log_file = osp.join(tsne_work_dir, 'extract.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    #dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    #dataset = build_dataset(dataset_cfg.data.extract)
    dataset = build_dataset(cfg.data.extract)
    # compress dataset, select that the label is less then max_num_class
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
        #samples_per_gpu=dataset_cfg.data.samples_per_gpu,
        #workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model
    # # tune the model
    # PATCH_SIZE = (10,10) # 10 for widar; try other for csi office # int or tuple(H,W)
    # # cfg.model = dict(
    # # type='Classification_CSI',
    # # img_like_tensor=True,
    # # backbone=dict(
    # #     type='ResNet',
    # #     depth=18,
    # #     in_channels=3,
    # #     out_indices=[4],  # 0: conv-1, x: stage-x
    # #     norm_cfg=dict(type='BN')),
    # # head=dict(type='ClsCsiHead', with_avg_pool=True, in_channels=512,
    # #     num_classes=22), # widar all 22; widar user123 6; OR 7; Signfi 276
    # # AE_model = False
    # # )
    # cfg.model = dict(
    # type='Classification_CSI_VIT',
    # backbone=dict(
    #     type='VisionTransformer_CSI',
    #     arch='csi-small',  # embed_dim = 384 # for widar 'b' is good enough; for office room we try csi-small
    #     img_size=(60,500), # (30,500) for widar;(30,2000) for csi office; (30,200) for signfi; (60,500) for widar_all
    #     patch_size=(10,10),  # 10 for widar; try other for csi office # int or tuple(H,W)
    #     stop_grad_conv1=True),
    # head=dict(
    #     type='ClsCsiHead',
    #     in_channels=768,
    #     num_classes=22, # widar all 22; widar user123 6; OR 7; Signfi 276
    #     vit_backbone=True,
    # ))
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
    # extractor = MultiExtractProcess(
    #     pool_type=args.pool_type, backbone='resnet50', layer_indices=layer_ind)
    extractor = ExtractProcess()
    features = extractor.extract(model, data_loader, distributed=distributed)
    labels = dataset.data_source.get_gt_labels()

    # # save features
    # mmcv.mkdir_or_exist(f'{tsne_work_dir}features/')
    # logger.info(f'Save features to {tsne_work_dir}features/')
    
    # output_file = f'{tsne_work_dir}features/{cfg.name}_labels.npy'
    # np.save(output_file,labels)
    # if distributed:
    #     rank, _ = get_dist_info()
    #     if rank == 0:
    #         for key, val in features.items():
    #             output_file = \
    #                 f'{tsne_work_dir}features/{cfg.name}_{key}.npy'
    #                 #f'{tsne_work_dir}features/{dataset_cfg.name}_{key}.npy'
    #             np.save(output_file, val)
    # else:
    #     for key, val in features.items():
    #         output_file = \
    #             f'{tsne_work_dir}features/{cfg.name}_{key}.npy'
    #             #f'{tsne_work_dir}features/{dataset_cfg.name}_{key}.npy'
    #         np.save(output_file, val)

    # build t-SNE model
    # if args.learning_rate <0:
    #     args.learning_rate = 'auto'
    # tsne_model = TSNE(
    #     n_components=args.n_components,
    #     perplexity=args.perplexity,
    #     early_exaggeration=args.early_exaggeration,
    #     learning_rate=args.learning_rate,
    #     n_iter=args.n_iter,
    #     n_iter_without_progress=args.n_iter_without_progress,
    #     init=args.init,
    #     random_state=args.tsne_seed
    #     )
    for n in (200,300,400):
        for d in (0.0, 0.1, 0.25, 0.5, 0.99):
            tsne_model = umap.UMAP(n_neighbors=n,min_dist=d)
            # run and get results
            mmcv.mkdir_or_exist(f'{tsne_work_dir}saved_pictures/')
            logger.info('Running t-SNE......')
            for key, val in features.items():
                result = tsne_model.fit_transform(val)
                res_min, res_max = result.min(0), result.max(0)
                res_norm = (result - res_min) / (res_max - res_min)
                plot_tsne(res_n = res_norm,labels=labels,workdir=tsne_work_dir,key=key,others=[n,d])
            logger.info(f'Saved results to {tsne_work_dir}saved_pictures/')



def plot_tsne(res_n,labels,workdir,key,others):
    CORLORS = ['#251abf', '#8d008b', '#a80058', '#a7002f', 
    '#972810', '#992f1e', '#9a362b', '#9a3d36', '#9d3c60', '#8a4b87', '#5f5ea2', '#006ea9']
    ACT = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(V)',
    'Draw-N(V)','Draw-N(H)','Draw-O(H)','Draw-Rectangle(H)','Draw-Triangle(H)','Draw-Zigzag(H)']
    ACT = [0,1,2,3,4,5]
    cmap = ListedColormap(CORLORS)

    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)#dont show the top axis
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    scatter = ax.scatter(
        res_n[:, 0],
        res_n[:, 1],
        alpha=0.9,
        s=15,
        c=labels,
        cmap='tab20')
    # add legends
    # temp,slabels=scatter.legend_elements()
    # slabels = ['Push&Pull','Sweep','Clap','Slide',
    # 'Draw-N(H)','Draw-O(H)','Draw-Rectangle(H)','Draw-Triangle(H)','Draw-Zigzag(H)']
    # legend1 = ax.legend(*(temp,slabels),
    #                  loc="best",frameon=False, title="Activity Classes",title_fontsize=20,alignment='left',prop={'size': 20})
    legend1 = ax.legend(*scatter.legend_elements(),
        loc="best",frameon=True, title="Activity Classes",title_fontsize=14,alignment='left',prop={'size': 14})
    
    ax.add_artist(legend1)
    # save fig
    fig.savefig(f'{workdir}saved_pictures/{key}n{others[0]}d{others[1]}.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    #plt.savefig(f'{workdir}saved_pictures/{key}n{others[0]}d{others[1]}.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)

if __name__ == '__main__':
    main()
