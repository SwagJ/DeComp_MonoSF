#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/disk_hdd/kitti_flow"
CHECKPOINT="/disk_ssd/Self_Mono_Experiments/-mono-flow-disp-warp-feat-norm-top-512-/checkpoint_epoch30.ckpt"

# model
MODEL=MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm

Valid_Dataset=KITTI_2015_Train_Full_mnsf
Valid_Augmentation=Augmentation_Resize_Only_512x768
Valid_Loss_Function=Eval_FlowDisp_KITTI_Train

# training configuration
SAVE_PATH="/disk_ssd/self-mono-eval/-mono-flow-disp-warp-feat-norm-top-512-"
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
--validation_dataset_preprocessing_crop=False \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key="f1" \
#--save_disp=True \
#--save_disp2=False \
#--save_flow=True \
#--save_flow_otl=True \
#--save_disp_otl=True




