# Decomposing Self-Supervised Monocular Scene Flow Estimation

This repository is the implementation of master thesis work: **Decomposing Self-Supervised Monocular Scene Flow Estimation** by Jingyuan Ma, supervised by [Dr. Martin Danelljan](https://martin-danelljan.github.io/) and [Dr. Radu Timofte](http://people.ee.ethz.ch/~timofter/). 

- Contact: majing[at]student.ethz.ch 

This implementation is based on the Self-Mono's implementation:
 

&nbsp;&nbsp;&nbsp;[**Self-Supervised Monocular Scene Flow Estimation**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hur_Self-Supervised_Monocular_Scene_Flow_Estimation_CVPR_2020_paper.pdf)    
&nbsp;&nbsp;&nbsp;*CVPR*, 2020 (**Oral**)  

The official implementaion of this paper can be found [here](https://github.com/visinf/self-mono-sf)

## Getting started
This code has been developed with Anaconda (Python 3.7), **PyTorch 1.2.0** and CUDA 10.1.243 on Ubuntu 19.10.  
Based on a fresh [Anaconda](https://www.anaconda.com/download/) distribution and [PyTorch](https://pytorch.org/) installation, following packages need to be installed:  

  ```Shell
  pip install tensorboard
  pip install pypng==0.0.18
  ```

Then, please excute the following to install the Correlation and Forward Warping layer:
  ```Shell
  ./install_modules.sh
  ```

**For PyTorch version > 1.3**  
Please put the **`align_corners=True`** flag in the `grid_sample` function in the following files:
  ```
  augmentations.py
  losses.py
  models/modules_sceneflow.py
  utils/sceneflow_util.py
  ```


## Dataset

Please download the following to datasets for the experiment:
  - [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) (synced+rectified data, please refer [MonoDepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) for downloading all data more easily)
  - [KITTI Scene Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

## Training and Inference
The **[scripts](scripts/)** folder contains training\/inference scripts of all experiments demonstrated in this master thesis (including ablation study).

**For training of FD-Net**, you can simply run the following script files:

| Script                                                                    |
|---------------------------------------------------------------------------|
| `./train_flow_disp_warp_og_decoder_feat_norm_census_600_final.sh`         | 


**For training of DM-Net**, you can simply run the following script files:  
| Script                           |
|----------------------------------|
| `./train_monoexp_v2_bb_kitti.sh` |

In the script files, please configure these following PATHs for experiments:
  - `EXPERIMENTS_HOME` : your own experiment directory where checkpoints and log files will be saved.
  - `KITTI_RAW_HOME` : the directory where *KITTI raw data* is located in your local system.
  - `KITTI_HOME` : the directory where *KITTI Scene Flow 2015* is located in your local system. 

Also, please drop the line 2 to line 13 if you are not using BIWI Cluster of CVL lab from ETHZ. 
   
  
**For testing the pretrained models**, you can simply run the following script files:

| Script                                          | Task                                      | Dataset          | 
|-------------------------------------------------|-------------------------------------------|------------------|
| `./eval_monoflowdisp_selfsup_kitti_train.sh`    | Joint Estimation of Optical Flow and Disp | KITTI 2015 Train |
| `./eval_monoflowexp_selfsup_kitti_train.sh`     | MonoSceneFlow                             | KITTI 2015 Train |

  
  - To save output image, please turn on `--save_disp=True`, `--save_disp2=True`, and `--save_flow=True` in the script.  

## Pretrained Models 

The **[checkpoints/DM_FD](checkpoints/DM_FD)** folder contains the checkpoints of the pretrained models of DM-Net or FD-Net.  

## Acknowledgement

Please cite our paper if you use our source code.  

- Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Self-Mono](https://github.com/visinf/self-mono-sf)  
- My sincerest gratitude to my two supervisors: [Dr. Martin Danelljan](https://martin-danelljan.github.io/) and [Dr. Radu Timofte](http://people.ee.ethz.ch/~timofter/)

