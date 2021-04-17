#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/disk_hdd/kitti_flow"
CHECKPOINT="/disk_ssd/Self_Mono_Experiments/-flowdisp-dispCv3-loss-3d-/checkpoint_best.ckpt"

# model
MODEL=MonoFlowDisp_DispC_v3

Valid_Dataset=KITTI_2015_Train_Full_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_MonoFlowDispC_KITTI_Train

# training configuration
SAVE_PATH="/disk_ssd/self-mono-eval/-flowdisp-dispCv3-loss-3d-/"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--backbone_mode=True \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_preprocessing_crop=False \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key="f1" \
--save_disp=False \
--save_disp2=True \
--save_flow=False \
--save_flow_otl=False \
--save_disp2_otl=True \
--save_disp_otl=False




