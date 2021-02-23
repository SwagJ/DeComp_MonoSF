from . import model_monosceneflow
from . import model_monosceneflow_ablation
from . import model_monosceneflow_ablation_decoder_split
from . import model_monodepth_ablation
from . import model_exp_depth_flow

##########################################################################################
## Monocular Scene Flow - The full model 
##########################################################################################

MonoSF_Full		=	model_monosceneflow.MonoSceneFlow
MonoSF_Disp_Exp		=	model_monosceneflow.MonoSF_Disp_Exp
MonoSF_Disp_Exp_Plus		=	model_monosceneflow.MonoSF_Disp_Exp_Plus
MonoSceneFlow_Disp_Res      =	model_monosceneflow.MonoSceneFlow_Disp_Res
MonoFlow_Disp 				=	model_monosceneflow.MonoFlow_Disp
Mono_Expansion				=   model_exp_depth_flow.Mono_Expansion
PWC_Disp					=   model_exp_depth_flow.PWC_Disp
PWC_Disp_Unfreeze			=	model_exp_depth_flow.PWC_Disp_Unfreeze
MonoFlow_Disp_Seperate_NoWarp = model_monosceneflow.MonoFlow_Disp_Seperate_NoWarp
MonoFlow_Disp_Seperate_Warp_OG_Decoder = model_monosceneflow.MonoFlow_Disp_Seperate_Warp_OG_Decoder
MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res = model_monosceneflow.MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res
MonoSceneFlow_PWC 			=	model_monosceneflow.MonoSceneFlow_PWC
MonoFlowExp_ppV1			=	model_monosceneflow.MonoFlowExp_ppV1
MonoFlowExp_ppV2			=	model_monosceneflow.MonoFlowExp_ppV2

##########################################################################################
## Monocular Scene Flow - The models for the ablation studies
##########################################################################################

MonoSceneFlow_CamConv			=	model_monosceneflow_ablation.MonoSceneFlow_CamConv

MonoSceneFlow_FlowOnly			=	model_monosceneflow_ablation.MonoSceneFlow_OpticalFlowOnly
MonoSceneFlow_DispOnly			=	model_monosceneflow_ablation.MonoSceneFlow_DisparityOnly

MonoSceneFlow_Split_Cont		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split_base
MonoSceneFlow_Split_Last1		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split1
MonoSceneFlow_Split_Last2		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split2
MonoSceneFlow_Split_Last3		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split3
MonoSceneFlow_Split_Last4		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split4
MonoSceneFlow_Split_Last5		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split5

##########################################################################################
## Monocular Depth - The models for the ablation study in Table 1. 
##########################################################################################

MonoDepth_Baseline				= model_monodepth_ablation.MonoDepth_Baseline
MonoDepth_CamConv				= model_monodepth_ablation.MonoDepth_CamConv