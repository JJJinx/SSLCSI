# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import numpy as np
import mmcv
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.cnn.utils.weight_init import trunc_normal_

from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import ExtractProcess,knn_classifier,accuracy,FocalLoss,balanced_softmax_loss
from mmselfsup.utils import get_root_logger
from mmselfsup.utils import (get_root_logger, multi_gpu_test,
                             setup_multi_processes, single_gpu_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSelfSup test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--clscheckpoint', help='checkpoint file of cls head')
    parser.add_argument('--scene_flag', help='checkpoint file of cls head')
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
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='(Deprecated, please use --local-rank)')
    parser.add_argument('--local-rank', type=int, default=0)
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
    # KNN settings
    parser.add_argument(
        '--num-knn',
        default=[1,2,3,4,5],
        nargs='+',
        type=int,
        help='Number of NN to use. 20 usually works the best.')
    parser.add_argument(
        '--temperature',
        default=0.07,
        type=float,
        help='Temperature used in the voting coefficient.')
    parser.add_argument(
        '--use-cuda',
        default=True,
        type=bool,
        help='Store the features on GPU. Set to False if you encounter OOM')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

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
    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'test_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the model and load checkpoint
    model = build_algorithm(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    model.eval()
    existence_clsfier = MAEMultilayerHead_CSI(embed_dim=2304,num_classes=2)
    if args.scene_flag == 'office':
        best_model_params_path =os.path.join(args.clscheckpoint,'existence_clsfier_office.pt')
        testlist = cfg.test_list
    else:
        best_model_params_path =os.path.join(args.clscheckpoint,'existence_clsfier_home.pt')
        testlist = cfg.test_list_home
    existence_clsfier.load_state_dict(torch.load(best_model_params_path)) #加载最优存在检测模型
    num_clsfier = MAEMultilayerHead_CSI(embed_dim=2304,num_classes=3) # TODO xuyao 重新定义训练集的label
    # build extraction processor and run
    extractor = ExtractProcess()

    # give the result for all score file

    for file in testlist:
        cfg.data.test.data_source.data_prefix = file
        # build the dataloader
        dataset_test = build_dataset(cfg.data.test)
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
        data_loader_test = build_dataloader(
            dataset_test,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        test_feats = extractor.extract(
            model, data_loader_test, distributed=distributed)['feat']
        test_feats = torch.from_numpy(test_feats) # 利用test feats输出标签，未打乱顺序

        
        # 写一个测试集合的dataloader 使用刚才的分类器进行预测 TODO 
        outputs = existence_clsfier(test_feats)
        _, preds = torch.max(outputs[0], 1)

        preds = np.array(preds)
        filename = file.split('/')[-2]+'_'+file.split('/')[-1].split('.')[0]+'.txt'
        np.savetxt(os.path.join(cfg.work_dir,filename),preds)
        # logger.info('\n',file,":") 
        # logger.info(preds) 

class MAEMultilayerHead_CSI(torch.nn.Module):
    def __init__(self, embed_dim, num_classes=1000,focal_loss_flag=True):
        super(MAEMultilayerHead_CSI, self).__init__()
        self.fc0 = nn.Linear(embed_dim, embed_dim//2)
        self.head = nn.Linear(embed_dim//2, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim//2, affine=False, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss_flag = focal_loss_flag
        self.criterion_focal = FocalLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.fc0.bias, 0)
        trunc_normal_(self.fc0.weight, std=0.01)

    def forward(self, x):
        """"Get the logits."""
        x = self.fc0(x)
        x = self.bn(x)
        x = self.relu(x)
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        if self.focal_loss_flag:
            #losses['loss'] = self.criterion_focal(outputs[0], labels)
            losses = balanced_softmax_loss(labels,outputs[0])
        else:
            losses= self.criterion(outputs[0], labels)
        return losses

if __name__ == '__main__':
    main()
