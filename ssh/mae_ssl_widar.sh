#!/bin/bash
#SBATCH --job-name=MAE         # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=33        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --account=su003-ihw1
#SBATCH --output=/output/log_stdout/mae-%A.out 
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=xuk16@uni.coventry.ac.uk

echo "Executing on the machine:" $(hostname)

module purge
module load GCC/11.2.0 Python/3.9.6 CUDA/11.3  OpenMPI/4.1.1

set -x

JOB_NAME='MAE'
CONFIG='/mmselfsup/configs/selfsup/mae/mae_vit-base-p10_8xb512-coslr-400e_csi.py'
CONFIG_VAL='/mmselfsup/configs/benchmarks/classification/csi_widar/sup_vit-b_1xb64-coslr-100e.py'
CONFIG_TEST='/mmselfsup/configs/benchmarks/classification/csi_widar/sup_vit-b_1xb64-coslr-100e.py'

SOURCE_DATA_DIR='/data/widar_all_r2/'
SOURCE_TRAIN_DATA='widar_r2_train.pt'
SOURCE_VAL_DATA='widar_r2_val.pt'
SOURCE_TEST_DATA='widar_r2_test.pt'

TARGET_DATA_DIR='/data/widar_all_r2/'
TARGET_TRAIN_DATA='widar_r2_train.pt'
TARGET_VAL_DATA='widar_r2_val.pt'
TARGET_TEST_DATA='widar_r2_test.pt'

TARGET2_DATA_DIR='/data/widar_all_r2/'
TARGET2_TRAIN_DATA='widar_r2_train.pt'
TARGET2_VAL_DATA='widar_r2_val.pt'
TARGET2_TEST_DATA='widar_r2_test.pt'


DATE=$(date "+%Y%m%d")
WORK_DIR='/mae_concat-lr2e-4-adamw-StepFixcoslr-400e_widaruser3room1r2_'${DATE}'/'

PORT='295'${SLURM_JOB_ID:0-2:2}

FINTUNE_FOLDER='2layer_room3_p1.0_'
FINTUNE_FOLDER2='2layer_user3_p1.0_'

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py ${CONFIG} \
        --cfg-options data.samples_per_gpu=64 \
        data.train.data_source.type='WiFi_Widar_amp_pt' \
        data.train.data_source.keep_antenna=True \
        data.train.data_source.multi_dataset=False \
        data.train.data_source.conj_pre=False \
        data.train.data_source.dual=False \
        data.train.data_source.data_prefix=${SOURCE_DATA_DIR}${SOURCE_TRAIN_DATA} \
        data.train.data_source.data_prefix_train=${SOURCE_DATA_DIR}${SOURCE_TRAIN_DATA} \
        data.train.data_source.data_prefix_val=${SOURCE_DATA_DIR}${SOURCE_VAL_DATA} \
        data.train.data_source.data_prefix_test=${SOURCE_DATA_DIR}${SOURCE_TEST_DATA} \
        data.train.data_source.prop=1.0 \
        model.type='MAE_CSI' \
        model.backbone.type='MAEViT_CSI' \
        model.backbone.in_channels=3 \
        model.backbone.arch='csi-s' \
        model.neck.in_chans=3 \
        model.neck.embed_dim=768 \
        model.head.type='MAEPretrainHead_CSI' \
        runner.max_epochs=250 \
        optimizer.lr=2e-3 \
        dist_params.port=$PORT \
        --work-dir=${WORK_DIR}'pretrain' --launcher="slurm"
# # epoch 250
# # model.type='MAE_Ant_CSI' \


python /mmselfsup/tools/model_converters/extract_backbone_weights.py \
        ${WORK_DIR}'pretrain/latest.pth' \
        ${WORK_DIR}'pretrain/model_latest.pth'

#p=1.0
###1 layer
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    data.train.data_source.type='WiFi_Widar_amp_pt' \
    data.val.data_source.type='WiFi_Widar_amp_pt' \
    data.train.data_source.keep_antenna=True \
    data.val.data_source.keep_antenna=True \
    data.train.data_source.conj_pre=False \
    data.val.data_source.conj_pre=False \
    data.train.data_source.dual=True \
    data.val.data_source.dual=True \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=33 \
    model.type='Classification_CSI_VIT' \
    model.backbone.type='VisionTransformer_CSI' \
    model.backbone.arch='csi-s' \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.finetune=False \
    model.backbone.in_channels=3 \
    model.head.type='MAELinprobeHead_CSI' \
    model.head.embed_dim=768 \
    lr_config.warmup_iters=5 \
    lr_config.warmup_ratio=0.0001 \
    optimizer.lr=2e-4 \
    runner.max_epochs=100 \
    find_unused_parameters=True \
    dist_params.port=$PORT \
    --work-dir=${WORK_DIR}'1layer_room2_p1.0_'${SLURM_JOB_ID} \
    --seed 0 \
    --launcher="slurm" ${PY_ARGS}

FTDIR=${WORK_DIR}'1layer_room2_p1.0_'${SLURM_JOB_ID}
for file in `find ${FTDIR} -name "*.pth"`; do
    MODEL_PTH=$file
done

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    ${MODEL_PTH} \
    --cfg-options data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TEST_DATA} \
    data.val.data_source.type='WiFi_Widar_amp_pt' \
    data.val.data_source.keep_antenna=True \
    data.val.data_source.conj_pre=False \
    data.val.data_source.dual=True \
    data.samples_per_gpu=33 \
    model.type='Classification_CSI_VIT' \
    model.backbone.type='VisionTransformer_CSI' \
    model.backbone.arch='csi-s' \
    model.backbone.in_channels=3 \
    model.head.type='MAELinprobeHead_CSI' \
    model.head.embed_dim=768 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'1layer_room2_p1.0_'${SLURM_JOB_ID} \
    --launcher="slurm" \


# ###2 layer
# p=1.0
srun --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --kill-on-bad-exit=1 \
     python -u /mmselfsup/tools/train.py $CONFIG_VAL \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    data.train.data_source.type='WiFi_Widar_amp_pt' \
    data.val.data_source.type='WiFi_Widar_amp_pt' \
    data.train.data_source.keep_antenna=True \
    data.val.data_source.keep_antenna=True \
    data.train.data_source.multi_dataset=False \
    data.train.data_source.conj_pre=False \
    data.train.data_source.dual=False \
    data.val.data_source.multi_dataset=False \
    data.val.data_source.conj_pre=False \
    data.val.data_source.dual=False \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=64 \
    model.type='Classification_CSI_VIT' \
    model.backbone.type='VisionTransformer_CSI' \
    model.backbone.in_channels=3 \
    model.backbone.arch='csi-s' \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.finetune=False \
    model.head.type='MAEMultilayerHead_CSI' \
    model.head.embed_dim=768 \
    lr_config.warmup_iters=5 \
    lr_config.warmup_ratio=0.0001 \
    optimizer.lr=1e-3 \
    runner.max_epochs=100 \
    find_unused_parameters=True \
    dist_params.port=$PORT \
    --work-dir=${WORK_DIR}${FINTUNE_FOLDER}${SLURM_JOB_ID} \
    --launcher="slurm" ${PY_ARGS}

FTDIR=${WORK_DIR}${FINTUNE_FOLDER}${SLURM_JOB_ID}
for file in `find ${FTDIR} -name "*.pth"`; do
    MODEL_PTH=$file
done


CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/analysis_tools/ACC_F1_confusion_matrix.py $CONFIG_TEST \
    ${MODEL_PTH} \
    --cfg-options data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TEST_DATA} \
    data.val.data_source.type='WiFi_Widar_amp_pt' \
    data.val.data_source.keep_antenna=True \
    data.val.data_source.multi_dataset=False \
    data.val.data_source.conj_pre=False \
    data.val.data_source.dual=False \
    data.samples_per_gpu=64 \
    model.type='Classification_CSI_VIT' \
    model.backbone.type='VisionTransformer_CSI' \
    model.backbone.arch='csi-s' \
    model.backbone.in_channels=3 \
    model.head.type='MAEMultilayerHead_CSI' \
    model.head.embed_dim=768 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}${FINTUNE_FOLDER}${SLURM_JOB_ID} \
    --launcher="slurm"
