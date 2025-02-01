#!/bin/bash

JOB_NAME='MoCov2'
CONFIG='/code/mmselfsup/configs/selfsup/mocov2/mocov2_resnet_1xb32-coslr-150e.py'
CONFIG_VAL='/code/mmselfsup/configs/benchmarks/classification/csi_office/sup_resnet_1xb64-coslr-100e.py'
CONFIG_TEST='/code/mmselfsup/configs/benchmarks/classification/csi_office/sup_resnet_1xb64-coslr-100e.py'
SOURCE_DATA_DIR='/data/wifiOfficeRoom/'
SOURCE_TRAIN_DATA='OR_train.pt'
TARGET_DATA_DIR='/data/wifiOfficeRoom/'
TARGET_TRAIN_DATA='OR_train.pt'
TARGET_VAL_DATA='OR_val.pt'
TARGET_TEST_DATA='OR_test.pt'
DATA_SOURCE='WiFi_Office_pt'


DATE=$(date "+%Y%m%d")
WORK_DIR='/output/mocov2_lr2e-5_resnet_adamw-coslr-150e_utharp100_'${DATE}'/'
PORT='29500'

FINTUNE_FOLDER='2layer_room3_p1.0_'
FINTUNE_FOLDER2='2layer_user3_p1.0_'

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py ${CONFIG} \
        --cfg-options data.samples_per_gpu=128 \
        data.train.data_source.type=${DATA_SOURCE} \
        data.train.data_source.keep_antenna=True \
        data.train.data_source.data_prefix=${SOURCE_DATA_DIR}${SOURCE_TRAIN_DATA} \
        data.train.data_source.prop=1.0 \
        model.type='MoCo_CSI' \
        model.backbone.type='ResNet' \
        model.backbone.in_channels=3 \
        model.head.type='ContrastiveHead' \
        optimizer.lr=2e-5 \
        runner.max_epochs=100 \
        dist_params.port=$PORT \
        --work-dir=${WORK_DIR}'pretrain' --launcher="pytorch"


python /code/mmselfsup/tools/model_converters/extract_backbone_weights.py \
        ${WORK_DIR}'pretrain/latest.pth' \
        ${WORK_DIR}'pretrain/model_latest.pth'

p=1.0
##1layer
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options data.train.data_source.type=${DATA_SOURCE} \
    data.val.data_source.type=${DATA_SOURCE} \
    data.train.data_source.keep_antenna=True \
    data.val.data_source.keep_antenna=True \
    data.train.data_source.multi_dataset=True \
    data.train.data_source.conj_pre=False \
    data.train.data_source.dual=False \
    data.val.data_source.multi_dataset=True \
    data.val.data_source.conj_pre=False \
    data.val.data_source.dual=False \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=128 \
    model.type='Classification_CSI_ResNet' \
    model.backbone.type='ResNet' \
    model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.in_channels=3 \
    model.backbone.frozen_stages=4 \
    model.head.type='ClsCsiHead' \
    model.head.in_channels=512 \
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
    data.val.data_source.type=${DATA_SOURCE} \
    data.val.data_source.keep_antenna=True \
    data.val.data_source.multi_dataset=False \
    data.val.data_source.conj_pre=True \
    data.val.data_source.dual=True \
    data.samples_per_gpu=128 \
    model.type='Classification_CSI_ResNet_Dual' \
    model.backbone.type='ResNet_Dual_CSI' \
    model.backbone.in_channels=2 \
    model.head.type='ClsCsiHead_Dual' \
    model.head.in_channels=512 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'1layer_room2_p1.0_' \
    --launcher="pytorch" \

###2layer
# p = 1.0
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options data.train.data_source.type=${DATA_SOURCE} \
    data.val.data_source.type=${DATA_SOURCE} \
    data.train.data_source.keep_antenna=True \
    data.val.data_source.keep_antenna=True \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=128 \
    model.type='Classification_CSI_ResNet' \
    model.backbone.type='ResNet' \
    model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.in_channels=3 \
    model.backbone.frozen_stages=4 \
    model.head.type='MultiCSIClsHead' \
    model.head.in_channels=512 \
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
    data.samples_per_gpu=128 \
    model.type='Classification_CSI_ResNet' \
    model.backbone.type='ResNet' \
    model.backbone.in_channels=3 \
    model.head.type='MultiCSIClsHead' \
    model.head.in_channels=512 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'2layer_room2_p1.0_' \
    --launcher="pytorch" \

