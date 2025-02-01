#!/bin/bash

JOB_NAME='MoCov2'
CONFIG='/code/mmselfsup/configs/selfsup/mocov2/mocov2_causualnet10_8xb32-coslr-200e_widar.py'
CONFIG_VAL='/code/mmselfsup/configs/benchmarks/classification/csi_widar/sup_causualnet_1xb64-coslr-100e.py'
CONFIG_TEST='/code/mmselfsup/configs/benchmarks/classification/csi_widar/sup_causualnet_1xb64-coslr-100e.py'
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
WORK_DIR='/output/mocov2_causual_sgd-coslr-150e_'${DATE}'/'
PORT='295'${pytorch_JOB_ID:0-2:2}

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py ${CONFIG} \
        --cfg-options data.samples_per_gpu=64 \
        data.train.data_source.type='WiFi_Widar_pt' \
        data.train.data_source.keep_antenna=False \
        data.train.data_source.multi_dataset=False \
        data.train.data_source.data_prefix=${SOURCE_DATA_DIR}${SOURCE_TRAIN_DATA} \
        model.backbone.in_channels=180 \
        optimizer.lr=2e-5 \
        runner.max_epochs=120 \
        find_unused_parameters=True \
        dist_params.port=$PORT \
        --work-dir=${WORK_DIR}'pretrain' --seed 0 --launcher="pytorch"

python /code/mmselfsup/tools/model_converters/extract_backbone_weights.py \
        ${WORK_DIR}'pretrain/latest.pth' \
        ${WORK_DIR}'pretrain/model_latest.pth'
        
#p=1.0

###1layer
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    data.train.data_source.type='WiFi_Widar_pt' \
    data.val.data_source.type='WiFi_Widar_pt' \
    data.train.data_source.keep_antenna=False \
    data.val.data_source.keep_antenna=False \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=32 \
    optimizer.lr=1e-3 \
    model.backbone.in_channels=180 \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.finetune=False \
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
    data.val.data_source.type='WiFi_Widar_pt' \
    data.val.data_source.keep_antenna=False \
    data.samples_per_gpu=32 \
    model.backbone.in_channels=180 \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'1layer_room2_p1.0_' \
    --launcher="pytorch" \

###2layer
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    /mmselfsup/tools/train.py  $CONFIG_VAL \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    data.train.data_source.type='WiFi_Widar_pt' \
    data.val.data_source.type='WiFi_Widar_pt' \
    data.train.data_source.keep_antenna=False \
    data.val.data_source.keep_antenna=False \
    data.train.data_source.prop=1.0 \
    data.train.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_TRAIN_DATA} \
    data.val.data_source.data_prefix=${TARGET_DATA_DIR}${TARGET_VAL_DATA} \
    data.samples_per_gpu=32 \
    optimizer.lr=1e-3 \
    model.backbone.in_channels=180 \
    model.backbone.init_cfg.checkpoint=${WORK_DIR}'pretrain/model_latest.pth' \
    model.backbone.finetune=False \
    model.head.type='MultiCSIClsHead' \
    runner.max_epochs=100 \
    find_unused_parameters=True \
    dist_params.port=$PORT \
    --work-dir=${WORK_DIR}'2layer_room2_p1.0_' \
    --seed 0 \
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
    data.val.data_source.type='WiFi_Widar_pt' \
    data.val.data_source.keep_antenna=False \
    data.samples_per_gpu=32 \
    model.backbone.in_channels=180 \
    model.head.type='MultiCSIClsHead' \
    dist_params.port=$PORT \
    --work-dir ${WORK_DIR}'2layer_room2_p1.0_' \
    --launcher="pytorch" \
