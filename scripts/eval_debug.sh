#!/bin/bash

# DATASETS_HOME
#KITTI_HOME="/disk_hdd/kitti_flow"
AUDI_HOME="/disk_hdd/a2d2/seq_01"
CHECKPOINT="./checkpoints/full_model_kitti/checkpoint_kitti_split.ckpt"

# model
MODEL=MonoSF_Full

Valid_Dataset=Audi_Test
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Test
#VALIDATION_AUGMENTATION_IMGSIZE=[400,1200]

# training configuration
SAVE_PATH="/disk_ssd/self-mono-eval/debug"
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
--validation_dataset_root=$AUDI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--save_disp=True \
--save_disp2=True \
--save_flow=True
