#!/bin/bash

# For SLURM cluster only
#SBATCH	--output=/scratch_net/phon/majing/src/log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --constraint='turing|titan_xp'

source /scratch_net/phon/majing/anaconda3/etc/profile.d/conda.sh
conda activate self-mono


# experiments and datasets meta
KITTI_RAW_HOME="/scratch_net/phon/majing/datasets/kitti_full/"
#KITTI_RAW_HOME="/disk_hdd/kitti_full/"
EXPERIMENTS_HOME="/scratch_net/phon/majing/src/exps"

# model
MODEL=MonoSF_Full

# save path

CHECKPOINT=None

# Loss and Augmentation
Train_Dataset=KITTI_Raw_KittiSplit_Train_mnsf
Train_Augmentation=Augmentation_SceneFlow
Train_Loss_Function=Loss_SceneFlow_SelfSup

Valid_Dataset=KITTI_Raw_KittiSplit_Valid_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Loss_SceneFlow_SelfSup

ALIAS="-self-mono-og-"
SAVE_PATH="$EXPERIMENTS_HOME/$ALIAS/"

#CHECKPOINT="$EXPERIMENTS_HOME/$ALIAS/checkpoint_latest.ckpt"





# training configuration
python ../main.py \
--batch_size=4 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--model=$MODEL \
--num_workers=10 \
--optimizer=Adam \
--optimizer_lr=2e-4 \
--save=$SAVE_PATH \
--total_epochs=62 \
--save_freq=8 \
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
