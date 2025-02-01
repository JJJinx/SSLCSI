#!/bin/bash

JOB_NAME='MAE'
CONFIG='/code/mmselfsup/configs/selfsup/mae/mae_vit-base-p10_8xb512-coslr-400e_uthar.py'
CONFIG_VAL='/code/mmselfsup/configs/benchmarks/classification/csi_office/sup_vit-b_1xb64-coslr-100e.py'
CONFIG_TEST='/code/mmselfsup/configs/benchmarks/classification/csi_office/sup_vit-b_1xb64-coslr-100e.py'
SOURCE_DATA_DIR='/data/wifiOfficeRoom/'
SOURCE_TRAIN_DATA='OR_train.pt'
TARGET_DATA_DIR='/data/wifiOfficeRoom/'
TARGET_TRAIN_DATA='OR_train.pt'
TARGET_VAL_DATA='OR_val.pt'
TARGET_TEST_DATA='OR_test.pt'
DATA_SOURCE='WiFi_Office_pt'


DATE=$(date "+%Y%m%d")
WORK_DIR='/output/work_dir/SSLBM/limited_unlabel/uthar/mae_utharp5_'${DATE}'/'

PORT='295'${pytorch_JOB_ID:0-2:2}

FINTUNE_FOLDER='2layer_room3_p1.0_'
FINTUNE_FOLDER2='2layer_user3_p1.0_'

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py ${CONFIG} \
    --cfg-options data.samples_per_gpu=64 \
    data.train.data_source.type=${DATA_SOURCE} \
    data.train.data_source.keep_antenna=True \
    data.train.data_source.data_prefix=${SOURCE_DATA_DIR}${SOURCE_TRAIN_DATA} \
    data.train.data_source.prop=0.05 \
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
    --work-dir=${WORK_DIR}'pretrain' --launcher="pytorch"
# # epoch 250
# # model.type='MAE_Ant_CSI' \

python /code/mmselfsup/tools/model_converters/extract_backbone_weights.py \
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
    --work-dir=${WORK_DIR}'1layer_room2_p1.0_' \
    --seed 0 \
    --launcher="pytorch" ${PY_ARGS}

FTDIR=${WORK_DIR}'1layer_room2_p1.0_'
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
    --work-dir ${WORK_DIR}'1layer_room2_p1.0_' \
    --launcher="pytorch" \

# ###2 layer
# p=1.0
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    data.train.data_source.type=${DATA_SOURCE} \
    data.val.data_source.type=${DATA_SOURCE} \
    data.train.data_source.keep_antenna=True \
    data.val.data_source.keep_antenna=True \
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
    --work-dir=${WORK_DIR}'2layer_room2_p1.0_' \
    --launcher="pytorch" ${PY_ARGS}

FTDIR=${WORK_DIR}'2layer_room2_p1.0_'
for file in `find ${FTDIR} -name "*.pth"`; do
    MODEL_PTH=$file
done


CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/analysis_tools/ACC_F1_confusion_matrix.py $CONFIG_TEST \
    ${MODEL_PTH} \
    --cfg-options data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TEST_DATA} \
    data.val.data_source.type=${DATA_SOURCE} \
    data.val.data_source.keep_antenna=True \
    data.samples_per_gpu=64 \
    model.type='Classification_CSI_VIT' \
    model.backbone.type='VisionTransformer_CSI' \
    model.backbone.arch='csi-s' \
    model.backbone.in_channels=3 \
    model.head.type='MAEMultilayerHead_CSI' \
    model.head.embed_dim=768 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'2layer_room2_p1.0_' \
    --launcher="pytorch"
