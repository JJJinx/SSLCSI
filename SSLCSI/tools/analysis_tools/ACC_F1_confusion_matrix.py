import argparse
import os
import os.path as osp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import f1_score

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls.core import calculate_confusion_matrix

from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.utils import (get_root_logger, multi_gpu_test,
                             setup_multi_processes, single_gpu_test)

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          file_name='confusion_matrix.png',
                          save_dir=None,
                          show=False,
                          title='Normalized Confusion Matrix',
                          color_theme='OrRd'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: False.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    # deal with those has no data
    per_label_sums[per_label_sums==0] = -1
    confusion_matrix = \
        np.abs(confusion_matrix / per_label_sums * 100)

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=300)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    colorbar = plt.colorbar(mappable=im, ax=ax)
    colorbar.ax.tick_params(labelsize=20)  # 设置 colorbar 标签的字体大小

    title_font = {'weight': 'bold', 'size': 20}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 40}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='k',
                size=20)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, file_name), format='png')
    if show:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSelfSup test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
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
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_algorithm(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    #record The number of test examples
    logger.info('Total number of samples in test dataset:{}'.format(len(dataset)))
    # record the test time
    start_time = time.time()
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: np.ndarray}
    end_time = time.time()
    inference_time_persample = (end_time - start_time)/len(dataset)
    logger.info('Total inference time:{}, inference time per sample{}'.format(end_time - start_time,inference_time_persample))
    
    
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     dataset.evaluate(outputs, logger, topk=(1,))


    # calculate all metric
    topk=(1,)
    for name, val in outputs.items(): #name is the name of the head
        val = torch.from_numpy(val)
        num = val.size(0)
        cls_num = val.size(1)
        target = torch.LongTensor(dataset.data_source.get_gt_labels())
        # val shape (batch_num,cls_num); target shape (batch_num)
        _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True) # pred is the predicted class not the score
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
        for k in (1,):
            correct_k = correct[:k].contiguous().view(-1).float().sum(
                0).item()
            acc = correct_k * 100.0 / num

        # original macro f1
        # tuned macro f1
        # weighted f1
        index_val = torch.argmax(val, dim=1).cpu().numpy()
        original_macro_f1 = f1_score(target,index_val,average='macro') *100
        #tuned_macro_f1 = cls_num*original_macro_f1/9
        original_micro_f1 = f1_score(target,index_val,average='micro') *100
        weighted_f1 = f1_score(target,index_val,average='weighted') *100
        print('\n results:    ')
        print('acc: {:.3f}'.format(acc))
        print('original_macro_f1: {:.3f}'.format(original_macro_f1))
        print('original_micro_f1: {:.3f}'.format(original_micro_f1))
        print('weighted_f1: {:.3f}'.format(weighted_f1))
        #confusion matrix
        cm = calculate_confusion_matrix(val,target) #matrix  [22,22]
        indice = [0, 1, 2, 3, 6, 7, 8, 9, 10]
        cm_filtered = cm[indice,:][:,indice]
        #df = pd.DataFrame(cm.numpy())
        #df.to_csv(osp.join(cfg.work_dir,'confusionMatrix.csv'), index=False)
        label_list = ['P&P','Sw','C','Sl','D-Z_V','D-N_V','D-N_H','D-O_H',
        'D-Rect_H','D-Tri_H','D-Z_H','D-O_V',
        'D-1','D-2','D-3','D-4','D-5',
        'D-6','D-7','D-8','D-9','D-0']
        plot_confusion_matrix(cm.numpy(),
                            label_list,
                            file_name='confusion_matrix.png',
                            save_dir=cfg.work_dir,
                            show=False,
                            title='Normalized Confusion Matrix',
                            color_theme='OrRd')
        label_list = [label_list[i] for i in indice]
        plot_confusion_matrix(cm_filtered.numpy(),
                            label_list,
                            file_name='confusion_matrix_filtered.png',
                            save_dir=cfg.work_dir,
                            show=False,
                            title='Normalized Confusion Matrix',
                            color_theme='OrRd')
if __name__ == '__main__':
    main()
