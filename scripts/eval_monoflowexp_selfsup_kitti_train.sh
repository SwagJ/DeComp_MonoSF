#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/disk_hdd/kitti_flow"
#AUDI_HOME="/disk_hdd/a2d2/seq_01"
CHECKPOINT="/disk_ssd/Self_Mono_Experiments/-monoexp-bb-/checkpoint_best.ckpt"
PRETRAIN="/disk_ssd/Self_Mono_Experiments/-mono-flow-disp-warp-og-decoder-no-res-/checkpoint_best.ckpt"

# model
MODEL=MonoDispExp

Valid_Dataset=KITTI_2015_Train_Full_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_MonoFlowExp_KITTI_Train
#VALIDATION_AUGMENTATION_IMGSIZE=[400,1200]

# training configuration
SAVE_PATH="/disk_ssd/self-mono-eval/-monoexp-bb-/"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--backbone_mode=True \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--save_disp=True \
--save_disp2=True \
--save_flow=True \
--save_flow_otl=True \
--save_disp_otl=True \
--save_disp2_otl=True
