import argparse
from multiprocessing.pool import RUN
import os.path as osp
from pdb import runeval
import time
from typing_extensions import runtime

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from sklearn.decomposition import PCA

from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import MultiExtractProcess,ExtractProcess
from mmselfsup.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='GKDE visualization')
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
    gkde_work_dir = osp.join(cfg.work_dir, f'gkde_{timestamp}/')#osp.join(cfg.work_dir, f'tsne_{timestamp}/')
    mmcv.mkdir_or_exist(osp.abspath(gkde_work_dir))
    log_file = osp.join(gkde_work_dir, 'extract.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
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
    #alignment hist (positive pair的距离计算的柱状图，需要专门写一个algorithm在extract部分实现输出这个值) + uniformity gkde TODO

    mmcv.mkdir_or_exist(f'{gkde_work_dir}saved_pictures/')
    logger.info('Running GKDE......')
    for key, val in features.items(): # key = feat, val.shape [N,1]
        val = val.squeeze()
        plot_hist(data = val,labels=labels,workdir=gkde_work_dir,key=key)
    # pca = PCA(n_components=2)
    # #print(features.items)
    # #raise RuntimeError
    # for key, val in features.items():
    #     pca.fit(val)
    #     pca_x = pca.transform(val)
    #     # 计算距离然后使用柱状图表示
    #     plot_tsne(data = pca_x,labels=labels,workdir=gkde_work_dir,key=key)
    #     logger.info(f'Saved results to {gkde_work_dir}saved_pictures/')
def plot_hist(data,labels,workdir,key):
    CORLORS = ['#251abf', '#8d008b', '#a80058', '#a7002f', 
    '#972810', '#992f1e', '#9a362b', '#9a3d36', '#9d3c60', '#8a4b87', '#5f5ea2', '#006ea9']
    ACT = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(V)',
    'Draw-N(V)','Draw-N(H)','Draw-O(H)','Draw-Rectangle(H)','Draw-Triangle(H)','Draw-Zigzag(H)']
    cmap = ListedColormap(CORLORS)

    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.spines["top"].set_visible(False)#dont show the top axis
    # ax.spines["right"].set_visible(True)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    
    ax.hist(data, bins=15, color='blue', edgecolor='black')
    # save fig
    fig.savefig(f'{workdir}saved_pictures/{key}.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)


def plot_gkde(data,labels,workdir,key):
    CORLORS = ['#251abf', '#8d008b', '#a80058', '#a7002f', 
    '#972810', '#992f1e', '#9a362b', '#9a3d36', '#9d3c60', '#8a4b87', '#5f5ea2', '#006ea9']
    ACT = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(V)',
    'Draw-N(V)','Draw-N(H)','Draw-O(H)','Draw-Rectangle(H)','Draw-Triangle(H)','Draw-Zigzag(H)']
    cmap = ListedColormap(CORLORS)

    # 使用Seaborn的kdeplot进行高斯核密度估计可视化
    sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap='viridis', fill=True, levels=20)
    #sns.scatterplot(x=data[:, 0], y=data[:, 1], color='blue', alpha=0.2)

    # 设置图形属性
    plt.title('2D Gaussian Kernel Density Estimation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # save fig
    plt.savefig(f'{workdir}saved_pictures/{key}.png')

    
if __name__ == '__main__':
    main()
