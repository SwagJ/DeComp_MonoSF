#!/bin/bash

# For SLURM cluster only
#SBATCH	--output=/scratch_net/phon/majing/src/log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

#source /scratch_net/phon/majing/anaconda/bin/conda shell.bash hook
#conda activate self-mono


# experiments and datasets meta
#KITTI_RAW_HOME="/scratch_net/phon/majing/datasets/kitti_full/"
KITTI_RAW_HOME="/disk_hdd/kitti_raw/"
EXPERIMENTS_HOME="/disk_ssd/self-mono-debug"
KITTI_COMB_HOME="/disk_hdd/kitti_full"
KITTI_FLOW_HOME="/disk_hdd/kitti_full/kitti_flow"
SYNTH_DRIVING_HOME="/disk_ssd/driving"

# model
MODEL=MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm 

# save path
#CHECKPOINT="checkpoints/full_model_kitti/checkpoint_latest.ckpt"
#CHECKPOINT="/disk_ssd/Self_Mono_Experiments/-monoflowdispC-v2-1-loss-v3/checkpoint_best.ckpt"
CHECKPOINT=None

# Loss and Augmentation
Train_Dataset=KITTI_Raw_KittiSplit_Train_mnsf
Train_Augmentation=Augmentation_SceneFlow_512x768
Train_Loss_Function=Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_Top

Valid_Dataset=KITTI_Raw_KittiSplit_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only_512x768
Valid_Loss_Function=Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_Top

ALIAS="-kitti-raw-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/debug"

PRETRAIN="/disk_ssd/Self_Mono_Experiments/-mono-flow-disp-warp-og-decoder-no-res-/checkpoint_epoch40.ckpt"


# training configuration
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[10, 32, 33, 34, 35, 36, 37, 38, 39, 40]" \
--backbone_weight=$PRETRAIN \
--backbone_mode=True \
--exp_training=False  \
--model=$MODEL \
--start=1000	\
--num_workers=10 \
--optimizer=Adam \
--optimizer_lr=5e-5 \
--save=$SAVE_PATH \
--total_epochs=62 \
--save_freq=5 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_RAW_HOME \
--training_dataset_flip_augmentations=True \
--training_dataset_preprocessing_crop=True \
--training_dataset_num_examples=-1 \
--training_key=total_loss \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_RAW_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=total_loss \
--validation_loss=$Valid_Loss_Function \