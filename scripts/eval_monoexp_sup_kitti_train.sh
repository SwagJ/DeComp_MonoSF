#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/disk_hdd/kitti_flow"
#AUDI_HOME="/disk_hdd/a2d2/seq_01"
CHECKPOINT="/disk_ssd/Self_Mono_Experiments/-monoexpansion-kitti-/checkpoint_epoch90.ckpt"

# model
MODEL=Mono_Expansion

Valid_Dataset=KITTI_2015_Train_Full_monoexp_eval
Valid_Augmentation=Augmentation_MonoExp_Eval_Only
Valid_Loss_Function=Eval_MonoExp_KITTI_Train
#VALIDATION_AUGMENTATION_IMGSIZE=[400,1200]

# training configuration
SAVE_PATH="/disk_ssd/self-mono-eval/-monoexpansion-kitti-/"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
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
#--save_disp=True \
#--save_disp2=True \
#--save_flow=True \
#--save_flow_otl=True \
#--save_disp2_otl=True
