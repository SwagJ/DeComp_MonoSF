#!/bin/bash

# For SLURM cluster only
#SBATCH	--output=/scratch_net/phon/majing/src/log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --constraint='turing'

#source /scratch_net/phon/majing/anaconda/bin/conda shell.bash hook
conda activate self-mono


# experiments and datasets meta
KITTI_RAW_HOME="/scratch_net/phon/majing/datasets/kitti_full/"
#KITTI_RAW_HOME="/disk_hdd/kitti_full/"
EXPERIMENTS_HOME="/scratch_net/phon/majing/src/debug"

# model
MODEL=MonoSF_Full

# save path
ALIAS="-eigen-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/"
CHECKPOINT="./checkpoints/full_model_kitti/checkpoint_latest.ckpt"

# Loss and Augmentation
Train_Dataset=KITTI_Comb_Train
Train_Augmentation=Augmentation_SceneFlow_Finetuning
Train_Loss_Function=Loss_SceneFlow_SemiSupFinetune

Valid_Dataset=KITTI_Comb_Val
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train

# training configuration
python ../main.py \
--batch_size=4 \
--batch_size_val=1 \
--finetuning=True \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[125, 187, 250, 281, 312]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=4e-5 \
--save=$SAVE_PATH \
--total_epochs=343 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_COMB_HOME \
--training_loss=$Train_Loss_Function \
--training_key=total_loss \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_COMB_HOME \
--validation_key=sf \
--validation_loss=$Valid_Loss_Function \
