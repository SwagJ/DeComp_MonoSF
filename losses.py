from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
from torchvision import transforms as vistf

from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow, flow2sf_dispC, flow2sf_dispC_v2, flow2sf_dispC_v3, flow2sf_depthC, flow2sf_dispC_validate
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing, pixel2pts_disp, disp2depth_kitti, flow2sf, flow2sf_exp, pts2pixel, pixel2pts
from utils.monodepth_eval import compute_errors, compute_d1_all
from models.modules_sceneflow import WarpingLayer_Flow, get_grid_exp


###############################################
## Basic Module 
###############################################

def _elementwise_epe(input_flow, target_flow):
	residual = target_flow - input_flow
	return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_l1(input_flow, target_flow):
	residual = target_flow - input_flow
	return torch.norm(residual, p=1, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
	residual = target_flow - input_flow
	return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _abs_robust_loss(diff, eps=0.01, q=0.4):
	return torch.pow((torch.abs(diff) + eps),q)

def _SSIM(x, y):
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	mu_x = nn.AvgPool2d(3, 1)(x)
	mu_y = nn.AvgPool2d(3, 1)(y)
	mu_x_mu_y = mu_x * mu_y
	mu_x_sq = mu_x.pow(2)
	mu_y_sq = mu_y.pow(2)

	sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
	sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
	sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

	SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
	SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
	SSIM = SSIM_n / SSIM_d

	SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

	return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)

def census_transform(image, patch_size):
	intensities = ((0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]) * 255).unsqueeze(1)
	kernel = torch.eye(patch_size * patch_size).view(patch_size * patch_size, 1, patch_size, patch_size).cuda()
	#print(kernel[1,0,:])
	neighbors = tf.conv2d(input=intensities, weight=kernel, stride=1, padding=patch_size//2)
	diff = neighbors - intensities
	# Coefficients adopted from DDFlow.
	diff_norm = diff / torch.sqrt(.81 + torch.pow(diff,2))

	return diff_norm

def soft_hamming(a_bhwk, b_bhwk, thresh=0.1):
	sq_dist_bhwk = torch.pow((a_bhwk - b_bhwk), 2)
	soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)

	return torch.sum(soft_thresh_dist_bhwk, dim=1, keepdim=True)

def census_loss(img0, img1, patch_size=7):
	census_image_a_bhwk = census_transform(img0, patch_size)
	census_image_b_bhwk = census_transform(img1, patch_size)

	hamming_bhw1 = soft_hamming(census_image_a_bhwk, census_image_b_bhwk)

	mask = torch.zeros_like(hamming_bhw1)
	border = patch_size // 2
	mask[:,:,border:-border,border:-border] = 1
	diff = _abs_robust_loss(hamming_bhw1) * mask.detach_()

	return diff, (mask > 0).detach_() 


def _apply_disparity(img, disp):
	batch_size, _, height, width = img.size()

	# Original coordinates of pixels
	x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
	y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

	# Apply shift in X direction
	x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
	flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
	# In grid_sample coordinates are assumed to be between -1 and 1
	output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

	return output

def _apply_disparity_features(features, disp):
	batch_size, _, height, width = features.size()

	# Original coordinates of pixels
	x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(features)
	y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(features)

	# Apply shift in X direction
	x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
	flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
	# In grid_sample coordinates are assumed to be between -1 and 1
	output_feat = tf.grid_sample(features, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

	return output_feat

def _generate_coords_left(disp):
	batch_size, _, height, width = disp.size()
	x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(disp)
	y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(disp)

	# Apply shift in X direction
	x_shifts = -disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
	flow_field = torch.stack((x_base + x_shifts, y_base), dim=3).permute(0,3,1,2)

	return flow_field

def _generate_image_left(img, disp):
	return _apply_disparity(img, -disp)

def _generate_feat_left(features,disp):
	return _apply_disparity_features(features, -disp)

def _adaptive_disocc_detection(flow):

	# init mask
	b, _, h, w, = flow.size()
	mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).float().requires_grad_(False)    
	flow = flow.transpose(1, 2).transpose(2, 3)

	disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
	disocc_map = (disocc > 0.5)

	if disocc_map.float().sum() < (b * h * w / 2):
		disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)
		
	return disocc_map

def _adaptive_disocc_detection_disp(disp):

	# # init
	b, _, h, w, = disp.size()
	mask = torch.ones(b, 1, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
	flow = torch.zeros(b, 2, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
	flow[:, 0:1, :, : ] = disp * w
	flow = flow.transpose(1, 2).transpose(2, 3)

	disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
	disocc_map = (disocc > 0.5)

	if disocc_map.float().sum() < (b * h * w / 2):
		disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)
		
	return disocc_map

def _gradient_x(img):
	img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
	gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
	return gx

def _gradient_y(img):
	img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
	gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
	return gy

def _gradient_x_2nd(img):
	img_l = tf.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
	img_r = tf.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
	gx = img_l + img_r - 2 * img
	return gx

def _gradient_y_2nd(img):
	img_t = tf.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
	img_b = tf.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
	gy = img_t + img_b - 2 * img
	return gy

def _smoothness_motion_2nd(sf, img, beta=1):
	sf_grad_x = _gradient_x_2nd(sf)
	sf_grad_y = _gradient_y_2nd(sf)

	img_grad_x = _gradient_x(img) 
	img_grad_y = _gradient_y(img) 
	weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
	weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

	smoothness_x = sf_grad_x * weights_x
	smoothness_y = sf_grad_y * weights_y

	return (smoothness_x.abs() + smoothness_y.abs())

def _smoothness_motion_2nd_alpha(sf, img, alpha = 1):
	sf_grad_x = _gradient_x_2nd(sf)
	sf_grad_y = _gradient_y_2nd(sf)

	img_grad_x = _gradient_x(img) 
	img_grad_y = _gradient_y(img) 
	weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x * alpha), 1, keepdim=True))
	weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y * alpha), 1, keepdim=True))

	smoothness_x = sf_grad_x * weights_x
	smoothness_y = sf_grad_y * weights_y

	return (smoothness_x.abs() + smoothness_y.abs())

def _disp2depth_kitti_K(disp, k_value): 

	mask = (disp > 0).float()
	depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))

	return depth

def _depth2disp_kitti_K(depth, k_value):

	disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

	return disp



###############################################
## Loss function
###############################################

class Loss_SceneFlow_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_SceneFlow_SelfSup_With_Consistency(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_With_Consistency, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)
		#print(reconstructImg(coord2, disp_l1).shape)

		depth_l1_warp = _disp2depth_kitti_K(reconstructImg(coord2, disp_l1), k_l2_aug[:,0,0])
		depth_l2_warp = _disp2depth_kitti_K(reconstructImg(coord1, disp_l2), k_l1_aug[:,0,0])

		depth_l1_warp = depth_l1_warp + tf.interpolate(sf_b, [h_dp, w_dp], mode="bilinear", align_corners=True)[:,2:3,:,:]
		depth_l2_warp = depth_l2_warp + tf.interpolate(sf_f, [h_dp, w_dp], mode="bilinear", align_corners=True)[:,2:3,:,:]

		disp_l1_warp = _depth2disp_kitti_K(depth_l1_warp, k_l2_aug[:,0,0])
		disp_l2_warp = _depth2disp_kitti_K(depth_l2_warp, k_l1_aug[:,0,0])

		disp_diff1 = _elementwise_l1(disp_l1 / w_dp, disp_l2_warp / w_dp).mean(dim=1, keepdim=True)
		disp_diff2 = _elementwise_l1(disp_l2 / w_dp, disp_l1_warp / w_dp).mean(dim=1, keepdim=True) 
		loss_consist1 = disp_diff1[occ_map_f].mean()
		loss_consist2 = disp_diff2[occ_map_b].mean()
		disp_diff1[~occ_map_f].detach_()
		disp_diff2[~occ_map_b].detach_()
		loss_consist = loss_consist1 + loss_consist2



		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s + self._sf_3d_pts * loss_consist
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_consist

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		loss_sf_cons = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_consist = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s
			loss_sf_cons = loss_sf_cons + loss_consist

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["s_cons"] = loss_sf_cons
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_SceneFlow_SelfSup_Depth3D(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_Depth3D, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._dp_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def depth_loss_3d(self,disp_l,disp_r, aug_size, k_l, k_r, ii):
		b, _, h,w = disp_l.size()
		disp_l = disp_l * w
		disp_r = disp_r * w

		#scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h
		local_scale[:, 1] = w

		# generate left & right coords
		warpped_r = _generate_coords_left(disp_l)
		warpped_l = _generate_coords_left(disp_r)

		# meshgrid
		grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
		grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

		ones = torch.ones_like(grid_h)
		pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()
		
		# intrinsic scaling
		rel_scale = local_scale / aug_size
		k_l_s = intrinsic_scale(k_l, rel_scale[:,0], rel_scale[:,1])
		k_r_s = intrinsic_scale(k_r, rel_scale[:,0], rel_scale[:,1])
		depth_l = disp2depth_kitti(disp_l, k_l_s[:, 0, 0])
		depth_r = disp2depth_kitti(disp_r, k_r_s[:, 0, 0])

		# generate 3d pts of left and right
		pts_l = pixel2pts_disp(k_l_s, depth_l, pixelgrid)
		pts_r = pixel2pts_disp(k_r_s, depth_r, pixelgrid)
		# generate warpped 3d pts of left and right
		#print(ones.shape)
		#print(warpped_l.shape)
		warpped_grid_r = torch.cat((warpped_r.cpu(),ones),dim=1).float().cuda()
		warpped_grid_l = torch.cat((warpped_l.cpu(),ones),dim=1).float().cuda()
		pts_warpped_l = pixel2pts_disp(k_r_s, depth_l, warpped_grid_l)
		pts_warpped_r = pixel2pts_disp(k_l_s, depth_r, warpped_grid_r)

		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		right_occ = _adaptive_disocc_detection_disp(disp_l).detach()

		pts_normed_l = torch.norm(pts_l, p=2, dim=1, keepdim=True)
		pts_normed_r = torch.norm(pts_r, p=2, dim=1, keepdim=True)

		pts_diff_l = _elementwise_epe(pts_l, pts_warpped_r).mean(dim=1, keepdim=True) / (pts_normed_l + 1e-8)
		pts_diff_r = _elementwise_epe(pts_r, pts_warpped_l).mean(dim=1, keepdim=True) / (pts_normed_r + 1e-8)

		loss_pts_l = pts_diff_l[left_occ].mean()
		loss_pts_r = pts_diff_r[right_occ].mean()

		pts_diff_l[~left_occ].detach_()
		pts_diff_r[~right_occ].detach_()

		return (loss_pts_r + loss_pts_r) * self._dp_3d_pts


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		loss_dp3d_sm = 0
		loss_dp2d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		k_r1_aug = target_dict['input_k_r1_aug']
		k_r2_aug = target_dict['input_k_r2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_disp3d_1 = self.depth_loss_3d(disp_l1,disp_r1, aug_size, k_l1_aug, k_r1_aug, ii)
			loss_disp3d_2 = self.depth_loss_3d(disp_l2,disp_r2, aug_size, k_l2_aug, k_r2_aug, ii)
			loss_dp2d_sm = loss_dp2d_sm + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
			loss_dp3d_sm = loss_dp3d_sm + (loss_disp3d_1 + loss_disp3d_2) * self._weights[ii]

			loss_dp_sum = loss_dp2d_sm + loss_dp3d_sm


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["dp_2d"] = loss_dp2d_sm
		loss_dict["dp_3d"] = loss_dp3d_sm
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		#loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_SceneFlow_SemiSupFinetune(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SemiSupFinetune, self).__init__()        

		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._unsup_loss = Loss_SceneFlow_SelfSup(args)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		unsup_loss_dict = self._unsup_loss(output_dict, target_dict)
		unsup_loss = unsup_loss_dict['total_loss']

		## Ground Truth
		gt_disp1 = target_dict['target_disp']
		gt_disp1_mask = (target_dict['target_disp_mask']==1).float()   
		gt_disp2 = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()   
		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		b, _, h_dp, w_dp = gt_disp1.size()     

		disp_loss = 0
		flow_loss = 0

		for ii, sf_f in enumerate(output_dict['flow_f_pp']):

			## disp1
			disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][ii], gt_disp1, mode="bilinear") * w_dp
			valid_abs_rel = torch.abs(gt_disp1 - disp_l1) * gt_disp1_mask
			valid_abs_rel[gt_disp1_mask == 0].detach_()
			disp_l1_loss = valid_abs_rel[gt_disp1_mask != 0].mean()

			## Flow Loss
			sf_f_up = interpolate2d_as(sf_f, gt_flow, mode="bilinear")
			out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], sf_f_up, disp_l1)
			valid_epe = _elementwise_robust_epe_char(out_flow, gt_flow) * gt_flow_mask
				
			valid_epe[gt_flow_mask == 0].detach_()
			flow_l1_loss = valid_epe[gt_flow_mask != 0].mean()

			## disp1_next
			out_depth_l1 = _disp2depth_kitti_K(disp_l1, target_dict['input_k_l1'][:, 0, 0])
			out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
			out_depth_l1_next = out_depth_l1 + sf_f_up[:, 2:3, :, :]
			disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, target_dict['input_k_l1'][:, 0, 0])

			valid_abs_rel = torch.abs(gt_disp2 - disp_l1_next) * gt_disp2_mask
			valid_abs_rel[gt_disp2_mask == 0].detach_()
			disp_l2_loss = valid_abs_rel[gt_disp2_mask != 0].mean()
			 
			disp_loss = disp_loss + (disp_l1_loss + disp_l2_loss) * self._weights[ii]
			flow_loss = flow_loss + flow_l1_loss * self._weights[ii]

			print(ii, disp_loss)

		# finding weight
		u_loss = unsup_loss.detach()
		d_loss = disp_loss.detach()
		f_loss = flow_loss.detach()

		max_val = torch.max(torch.max(f_loss, d_loss), u_loss)

		u_weight = max_val / u_loss
		d_weight = max_val / d_loss 
		f_weight = max_val / f_loss 

		total_loss = unsup_loss * u_weight + disp_loss * d_weight + flow_loss * f_weight
		loss_dict["unsup_loss"] = unsup_loss
		loss_dict["dp_loss"] = disp_loss
		loss_dict["fl_loss"] = flow_loss
		loss_dict["total_loss"] = total_loss

		return loss_dict



###############################################
## Eval
###############################################

def eval_module_disp_depth(gt_disp, gt_disp_mask, output_disp, gt_depth, output_depth):
	
	loss_dict = {}
	batch_size = gt_disp.size(0)
	gt_disp_mask_f = gt_disp_mask.float()

	## KITTI disparity metric
	d_valid_epe = _elementwise_epe(output_disp, gt_disp) * gt_disp_mask_f
	d_outlier_epe = (d_valid_epe > 3).float() * ((d_valid_epe / gt_disp) > 0.05).float() * gt_disp_mask_f
	loss_dict["otl"] = (d_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
	loss_dict["otl_img"] = d_outlier_epe

	## MonoDepth metric
	abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_disp_mask], output_depth[gt_disp_mask])        
	loss_dict["abs_rel"] = abs_rel
	loss_dict["sq_rel"] = sq_rel
	loss_dict["rms"] = rms
	loss_dict["log_rms"] = log_rms
	loss_dict["a1"] = a1
	loss_dict["a2"] = a2
	loss_dict["a3"] = a3

	return loss_dict


class Eval_MonoDepth_Eigen(nn.Module):
	def __init__(self):
		super(Eval_MonoDepth_Eigen, self).__init__()

	def forward(self, output_dict, target_dict):
		
		loss_dict = {}

		## Depth Eval
		gt_depth = target_dict['target_depth']

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_depth, mode="bilinear") * gt_depth.size(3)
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, target_dict['input_k_l1'][:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_mask = (gt_depth > 1e-3) * (gt_depth < 80)        

		## Compute metrics
		abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_depth_mask], out_depth_l1[gt_depth_mask])

		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1
		loss_dict["ab_r"] = abs_rel
		loss_dict["sq_r"] = sq_rel
		loss_dict["rms"] = rms
		loss_dict["log_rms"] = log_rms
		loss_dict["a1"] = a1
		loss_dict["a2"] = a2
		loss_dict["a3"] = a3

		return loss_dict


class Eval_SceneFlow_KITTI_Test(nn.Module):
	def __init__(self):
		super(Eval_SceneFlow_KITTI_Test, self).__init__()

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		##################################################
		## Depth 1
		##################################################
		input_l1 = target_dict['input_l1']
		intrinsics = target_dict['input_k_l1']

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], input_l1, mode="bilinear") * input_l1.size(3)
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		output_dict["out_disp_l_pp"] = out_disp_l1

		##################################################
		## Optical Flow Eval
		##################################################
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], input_l1, mode="bilinear")
		out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])        
		output_dict["out_flow_pp"] = out_flow

		##################################################
		## Depth 2
		##################################################
		out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next        

		loss_dict['sf'] = (out_disp_l1_next * 0).sum()

		return loss_dict


class Eval_SceneFlow_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_SceneFlow_KITTI_Train, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		#out_disp_l1 = target_dict['disp_pre'].cuda()
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print(output_dict['flow_f_pp'][0])
		
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		#out_disp_l1_next = interpolate2d_as(output_dict["disp_l2_pp"][0], gt_disp, mode="bilinear") * width
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class Eval_SceneFlow_3D(nn.Module):
	def __init__(self, args):
		super(Eval_SceneFlow_3D, self).__init__()
		self.warping_layer = WarpingLayer_Flow()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		#out_disp_l1 = target_dict['disp_pre'].cuda()
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		#dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)

		##################################################
		## Optical Flow Eval
		##################################################
		#print(output_dict['flow_f_pp'][0])
		
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		gt_pts1, _ = pixel2pts(target_dict['input_k_l1'], gt_depth_l1)
		pts1, _ = pixel2pts(target_dict['input_k_l1'], out_depth_l1)


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		#out_disp_l1_next = interpolate2d_as(output_dict["disp_l2_pp"][0], gt_disp, mode="bilinear") * width
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		#dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)

		gt_pts1_next, _ = pixel2pts(target_dict['input_k_l1'], gt_depth_l1_next)
		pts1_next, _ = pixel2pts(target_dict['input_k_l1'], out_depth_l1_next)

		gt_pts_warpped = self.warping_layer(gt_pts1_next, gt_flow)
		gt_sf = gt_pts_warpped - gt_pts1
		epe = _elementwise_epe(out_sceneflow, gt_sf) * gt_sf_mask
		loss_dict['sf_epe'] = torch.sum(epe) / torch.sum(gt_sf_mask.float())
		print((torch.norm(gt_sf,dim=1,keepdim=True)*gt_sf_mask).mean())
		print((torch.norm(out_sceneflow,dim=1,keepdim=True)*gt_sf_mask).mean())
		#print(gt_sf.shape)

		return loss_dict

class Eval_SceneFlow_KITTI_Train_Param(nn.Module):
	def __init__(self, args):
		super(Eval_SceneFlow_KITTI_Train_Param, self).__init__()
		self.warping_layer = WarpingLayer_Flow()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		#out_disp_l1 = target_dict['disp_pre'].cuda()
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		#dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		#output_dict["out_disp_l_pp"] = out_disp_l1
		#output_dict["out_depth_l_pp"] = out_depth_l1

		#d0_outlier_image = dict_disp0_occ['otl_img']
		#loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		#loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		#loss_dict["d1"] = dict_disp0_occ['otl']
		#output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print(output_dict['flow_f_pp'][0])
		
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, out_disp_l1)

		## Flow Eval
		# valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		# loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		# output_dict["out_flow_pp"] = out_flow

		# flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		# flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		# loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		# output_dict["otl_flow"] = flow_outlier_epe
		depth2 = out_depth_l1 + out_sceneflow[:,2:3,:,:]
		pts0, _ = pixel2pts(intrinsics, out_depth_l1)
		pts1, _ = pixel2pts(intrinsics, depth2)
		#print(pts1)
		pts1_warp = self.warping_layer(pts1, out_flow)
		recoverd_sf = pts1_warp - pts0
		#print(out_sceneflow[:,0:1,200:210,200:210])
		#print(recoverd_sf[:,0:1,200:210,200:210])
		# print(out_flow[:,1:2,:10,:10])
		#print((out_flow[:,0,:,:] / intrinsics[:,0,0].unsqueeze(0).unsqueeze(0).unsqueeze(0) * (out_depth_l1))[:,:10,:10])
		#print(out_sceneflow[:,0,:10,:10])
		#print(_disp2depth_kitti_K(dispC * width, intrinsics[:,0,0])[:,0,:10,:10])
		#print(recoverd_sf[:,2,:10,:10])
		#print(dispC[:,0,:10,:10])
		#print(_elementwise_l1(recoverd_sf[:, 0:1, :, :], out_sceneflow[:, 0:1, :, :]).shape)
		err_x = ((_elementwise_l1(recoverd_sf[:, 0:1, :, :], out_sceneflow[:, 0:1, :, :]))*gt_sf_mask).mean()
		err_y = ((_elementwise_l1(recoverd_sf[:, 1:2, :, :], out_sceneflow[:, 1:2, :, :]))*gt_sf_mask).mean()
		err_z = ((_elementwise_l1(recoverd_sf[:, 2:3, :, :], out_sceneflow[:, 2:3, :, :]))*gt_sf_mask).mean()
		#err_z = _elementwise_l1(recoverd_sf[:, 2:3, :, :], out_sceneflow[:, 2:3, :, :]).mean()

		recovered_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], recoverd_sf, out_disp_l1)
		# print(out_flow[:,0,:10,:10])
		# print(recovered_flow[:,0,:10,:10])
		# print(out_flow[:,0,200:210,200:210] - recovered_flow[:,0,200:210,200:210])

		loss_dict['err_x'] = err_x
		loss_dict['err_y'] = err_y
		loss_dict['err_z'] = err_z
		loss_dict['pts_err'] = (_elementwise_l1(pts1_warp, pts0) * gt_sf_mask).mean()
		loss_dict['flow_err'] = (_elementwise_epe(recovered_flow, out_flow) * gt_sf_mask).mean()

		# valid_epe = _elementwise_epe(recovered_flow, gt_flow) * gt_flow_mask
		# loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		# output_dict["out_flow_pp"] = out_flow

		# flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		# flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		# loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		#output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		# out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
		# out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		# #out_disp_l1_next = interpolate2d_as(output_dict["disp_l2_pp"][0], gt_disp, mode="bilinear") * width
		# gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		# dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		# output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		# output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		# d1_outlier_image = dict_disp1_occ['otl_img']
		# loss_dict["d2"] = dict_disp1_occ['otl']
		# output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		# outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		# loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class Eval_MonoFlowDispExp_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_MonoFlowDispExp_KITTI_Train, self).__init__()


	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		#out_disp_l1 = target_dict['disp_pre'].cuda()
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print(output_dict['flow_f_pp'][0])
		
		out_flow = self.upsample_flow_as(output_dict['flow_f_pp'][0], gt_flow)
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		disp_l1_next = output_dict["disp_l1_pp"][0] * (1 - torch.exp(output_dict["exp_f_pp"][0]))
		disp_l1_next = interpolate2d_as(disp_l1_next, gt_disp, mode="bilinear") * width

		out_depth_l1_next = _disp2depth_kitti_K(disp_l1_next, intrinsics[:, 0, 0])
		#out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class Eval_MonoFlowDispC_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_MonoFlowDispC_KITTI_Train, self).__init__()


	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		#out_disp_l1 = target_dict['disp_pre'].cuda()
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print(output_dict['flow_f_pp'][0])
		
		out_flow = self.upsample_flow_as(output_dict['flow_f_pp'][0], gt_flow)
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		#print(torch.max(gt_flow),torch.min(gt_flow))
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		disp_l1_next = output_dict["disp_l1_pp"][0] + output_dict["dispC_f_pp"][0]
		disp_l1_next = interpolate2d_as(disp_l1_next, gt_disp, mode="bilinear") * width

		out_depth_l1_next = _disp2depth_kitti_K(disp_l1_next, intrinsics[:, 0, 0])
		#out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class Eval_SceneFlow_KITTI_Train_Warpping(nn.Module):
	def __init__(self, args):
		super(Eval_SceneFlow_KITTI_Train_Warpping, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['sf_l'].cuda()
		gt_flow_mask = (target_dict['valid_sf_l']==1).float()

		#gt_disp = target_dict['target_disp']
		#gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		#gt_disp2_occ = target_dict['target_disp2_occ']
		#gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		#gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_flow.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_flow, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		#out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		#gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		#dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		#d0_outlier_image = dict_disp0_occ['otl_img']
		#loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		#loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		#loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = out_disp_l1

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		out_flow = out_flow.squeeze(0).cpu().data.numpy().transpose(1,2,0)
		gt_flow = gt_flow.squeeze(0).cpu().data.numpy().transpose(1,2,0)
		gt_flow_mask = gt_flow_mask.squeeze(0).cpu().data.numpy().transpose(1,2,0)

		#print(out_flow.shape, gt_flow.shape,gt_flow_mask.shape)
		valid_epe = (np.linalg.norm(out_flow - gt_flow[:,:,:2],axis=2) * gt_flow_mask[:,:,0]).mean()
		loss_dict["f_epe"] = valid_epe
		output_dict["out_flow_pp"] = out_flow

		#flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		#flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		#loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = valid_epe


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = out_depth_l1
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		#gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		#dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		#d1_outlier_image = dict_disp1_occ['otl_img']
		#loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = out_depth_l1_next


		##################################################
		## Scene Flow Eval
		##################################################

		#outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		#loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict


class Eval_Flow_KITTI_Train_Warpping(nn.Module):
	def __init__(self, args):
		super(Eval_Flow_KITTI_Train_Warpping, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['sf_l'].cuda()
		gt_flow_mask = (target_dict['valid_sf_l']==1).float()

		#gt_disp = target_dict['target_disp']
		#gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		#gt_disp2_occ = target_dict['target_disp2_occ']
		#gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		#gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_flow.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_flow, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		#out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		#gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		#dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		#d0_outlier_image = dict_disp0_occ['otl_img']
		#loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		#loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		#loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = out_disp_l1

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_flow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		out_flow = out_flow.squeeze(0).cpu().data.numpy().transpose(1,2,0)
		gt_flow = gt_flow.squeeze(0).cpu().data.numpy().transpose(1,2,0)
		gt_flow_mask = gt_flow_mask.squeeze(0).cpu().data.numpy().transpose(1,2,0)

		#print(out_flow.shape, gt_flow.shape,gt_flow_mask.shape)
		valid_epe = (np.linalg.norm(out_flow - gt_flow[:,:,:2],axis=2) * gt_flow_mask[:,:,0]).mean()
		loss_dict["f_epe"] = valid_epe
		output_dict["out_flow_pp"] = out_flow

		#flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		#flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		#loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = valid_epe


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = out_depth_l1
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		#gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		#dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		#d1_outlier_image = dict_disp1_occ['otl_img']
		#loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = out_depth_l1_next


		##################################################
		## Scene Flow Eval
		##################################################

		#outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		#loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class GT_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(GT_KITTI_Train, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = gt_disp
		output_dict["out_depth_l_pp"] = gt_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = gt_flow * gt_flow_mask

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = gt_disp2_occ
		output_dict["out_depth_l_pp_next"] = gt_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

##############################################
# Eval for flow and disp
##############################################
class Eval_FlowDisp_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_FlowDisp_KITTI_Train, self).__init__()

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_flow = self.upsample_flow_as(output_dict['flow_f_pp'][0], gt_flow)
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe

		output_dict["out_disp_l_pp_next"] = out_disp_l1

		return loss_dict


class Eval_FlowDisp_TS_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_FlowDisp_TS_KITTI_Train, self).__init__()

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["student_dict"]["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_flow = self.upsample_flow_as(output_dict['student_dict']['flow_f_pp'][0], gt_flow)
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe

		output_dict["out_disp_l_pp_next"] = out_disp_l1

		return loss_dict


class Eval_PWCDisp_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_PWCDisp_KITTI_Train, self).__init__()

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print("output_dict flow:", output_dict['flows_l'][-1].shape)
		
		out_flow = interpolate2d_as(output_dict['flows_l'][-1]*20, gt_flow)
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe

		output_dict["out_disp_l_pp_next"] = out_disp_l1

		return loss_dict

##########################################
# Evaluate MonoExp
##########################################
class Eval_MonoExp_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_MonoExp_KITTI_Train, self).__init__()

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1'].cuda() 
		out_disp_l1 = target_dict['disp0_pre'].cuda()               

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()
		#print(output_dict["dchange_f"].shape)

		#out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = gt_disp
		output_dict["out_depth_l_pp"] = gt_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']

		##################################################
		## Optical Flow Eval
		##################################################
		#print(gt_flow.shape)
		#print(target_dict['aug_size'])
		#print(gt_flow.size()[3] / target_dict['aug_size'][:,1])
		
		out_flow = interpolate2d_as(output_dict['flows_f'][-1]*20, gt_flow, mode="bilinear")
		out_flow[:,0:1,:,:] *= gt_flow.size()[3] / target_dict['aug_size'][:,1]
		out_flow[:,1:2,:,:] *= gt_flow.size()[2] / target_dict['aug_size'][:,0]
		#interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		dchange_f = output_dict["dchange_f"]
		dchange_f = interpolate2d_as(dchange_f,gt_disp2_occ)

		out_disp_l1_next = out_disp_l1 / torch.exp(dchange_f)
		out_depth_l1_next = _disp2depth_kitti_K(out_disp_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

class Eval_MonoFlowExp_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_MonoFlowExp_KITTI_Train, self).__init__()

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1'].cuda() 
		#out_disp_l1 = target_dict['disp0_pre'].cuda()               

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()
		#print(output_dict["dchange_f"].shape)

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']
		output_dict["otl_disp"] = d0_outlier_image

		##################################################
		## Optical Flow Eval
		##################################################
		#print(gt_flow.shape)
		#print(target_dict['aug_size'])
		#print(gt_flow.size()[3] / target_dict['aug_size'][:,1])
		
		out_flow = self.upsample_flow_as(output_dict['scaled_flow_f'], gt_flow)
		#interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["otl_flow"] = flow_outlier_epe


		##################################################
		## Depth 2
		##################################################

		dchange_f = output_dict["dchange_f"]
		dchange_f = interpolate2d_as(dchange_f,gt_disp2_occ)

		out_disp_l1_next = out_disp_l1 / torch.exp(dchange_f)
		out_depth_l1_next = _disp2depth_kitti_K(out_disp_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']
		output_dict["otl_disp2"] = d1_outlier_image


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d1_outlier_image.bool() + d0_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict

##############################################
# Eval for flow exp times depth network
##############################################
class Eval_Flow_ExpTimesDepth_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_Flow_ExpTimesDepth_KITTI_Train, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_flow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = output_dict["disp_l1_pp"][0] * output_dict['exp_f_pp'][0]
		out_depth_l1_next = interpolate2d_as(out_depth_l1_next, gt_disp2_occ, mode="bilinear")
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict


##############################################
# Eval for flow exp times depth network
##############################################
class Eval_Flow_ExpPlusDepth_KITTI_Train(nn.Module):
	def __init__(self, args):
		super(Eval_Flow_ExpPlusDepth_KITTI_Train, self).__init__()


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		gt_flow = target_dict['target_flow']
		gt_flow_mask = (target_dict['target_flow_mask']==1).float()

		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1).float()

		gt_disp2_occ = target_dict['target_disp2_occ']
		gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

		gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

		intrinsics = target_dict['input_k_l1']                

		##################################################
		## Depth 1
		##################################################

		batch_size, _, _, width = gt_disp.size()

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
		
		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		d0_outlier_image = dict_disp0_occ['otl_img']
		loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
		loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
		loss_dict["d1"] = dict_disp0_occ['otl']

		##################################################
		## Optical Flow Eval
		##################################################
		
		out_flow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
		#out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

		## Flow Eval
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
		loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		output_dict["out_flow_pp"] = out_flow

		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


		##################################################
		## Depth 2
		##################################################

		out_depth_l1_next = output_dict["disp_l1_pp"][0] + output_dict['exp_f_pp'][0]
		out_depth_l1_next = interpolate2d_as(out_depth_l1_next, gt_disp2_occ, mode="bilinear")
		out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
		gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

		dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
		
		output_dict["out_disp_l_pp_next"] = out_disp_l1_next
		output_dict["out_depth_l_pp_next"] = out_depth_l1_next

		d1_outlier_image = dict_disp1_occ['otl_img']
		loss_dict["d2"] = dict_disp1_occ['otl']


		##################################################
		## Scene Flow Eval
		##################################################

		outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
		loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

		return loss_dict



###############################################
## Ablation - Loss_SceneFlow_SelfSup
###############################################

class Loss_SceneFlow_SelfSup_NoOcc(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_NoOcc, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200


	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		# left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss: 
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = img_diff.mean()
		# loss_img = (img_diff[left_occ]).mean()
		# img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth#, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		## Depth2Pts
		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		# occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		# occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		# img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
		# img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1.mean()
		loss_im2 = img_diff2.mean()
		# loss_im1 = img_diff1[occ_map_f].mean()
		# loss_im2 = img_diff2[occ_map_b].mean()
		# img_diff1[~occ_map_f].detach_()
		# img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point Reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1.mean()
		loss_pts2 = pts_diff2.mean()
		# loss_pts1 = pts_diff1[occ_map_f].mean()
		# loss_pts2 = pts_diff2[occ_map_b].mean()
		# pts_diff1[~occ_map_f].detach_()
		# pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		## SceneFlow Loss
		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Depth Loss
			loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_SceneFlow_SelfSup_NoPts(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_NoPts, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200


	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss: 
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		## Depth2Pts
		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		# pts2_warp = reconstructPts(coord1, pts2)
		# pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		# img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
		# img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		# ## Point Reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		# pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		# pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		# loss_pts1 = pts_diff1[occ_map_f].mean()
		# loss_pts2 = pts_diff2[occ_map_b].mean()
		# pts_diff1[~occ_map_f].detach_()
		# pts_diff2[~occ_map_b].detach_()
		# loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s# + self._sf_3d_pts * loss_pts
		
		return sceneflow_loss, loss_im, loss_3d_s#, loss_pts

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		## SceneFlow Loss
		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		# loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Depth Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			# loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		# loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_SceneFlow_SelfSup_NoPtsNoOcc(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_NoPtsNoOcc, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200


	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		# left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss: 
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = img_diff.mean()
		# loss_img = (img_diff[left_occ]).mean()
		# img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth#, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		## Depth2Pts
		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		# pts2_warp = reconstructPts(coord1, pts2)
		# pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		# occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		# occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		# img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
		# img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1.mean()
		loss_im2 = img_diff2.mean()
		# loss_im1 = img_diff1[occ_map_f].mean()
		# loss_im2 = img_diff2[occ_map_b].mean()
		# img_diff1[~occ_map_f].detach_()
		# img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point Reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		# pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		# pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		# loss_pts1 = pts_diff1.mean()
		# loss_pts2 = pts_diff2.mean()
		# loss_pts1 = pts_diff1[occ_map_f].mean()
		# loss_pts2 = pts_diff2[occ_map_b].mean()
		# pts_diff1[~occ_map_f].detach_()
		# pts_diff2[~occ_map_b].detach_()
		# loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s # + self._sf_3d_pts * loss_pts
		
		return sceneflow_loss, loss_im, loss_3d_s # , loss_pts

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		## SceneFlow Loss
		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		# loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Depth Loss
			loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			# loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		# loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

#########################################################################
#
# Ablation on no image loss
#
#########################################################################

class Loss_SceneFlow_SelfSup_NoImg(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_NoImg, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		#img_l2_warp = reconstructImg(coord1, img_l2_aug)
		#img_l1_warp = reconstructImg(coord2, img_l1_aug)

		#img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#loss_im1 = img_diff1[occ_map_f].mean()
		#loss_im2 = img_diff2[occ_map_b].mean()
		#img_diff1[~occ_map_f].detach_()
		#img_diff2[~occ_map_b].detach_()
		#loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		#loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			#loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		#loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


###############################################
## Ablation - Separate Decoder
###############################################

class Loss_Flow_Only(nn.Module):
	def __init__(self):
		super(Loss_Flow_Only, self).__init__()

		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._warping_layer = WarpingLayer_Flow()

	def forward(self, output_dict, target_dict):

		## Loss
		total_loss = 0
		loss_sf_2d = 0
		loss_sf_sm = 0

		for ii, (sf_f, sf_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'])):

			## Depth2Pts            
			img_l1 = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2 = interpolate2d_as(target_dict["input_l2_aug"], sf_b)

			img_l2_warp = self._warping_layer(img_l2, sf_f)
			img_l1_warp = self._warping_layer(img_l1, sf_b)
			occ_map_f = _adaptive_disocc_detection(sf_b).detach()
			occ_map_b = _adaptive_disocc_detection(sf_f).detach()

			img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
			img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
			loss_im1 = img_diff1[occ_map_f].mean()
			loss_im2 = img_diff2[occ_map_b].mean()
			img_diff1[~occ_map_f].detach_()
			img_diff2[~occ_map_b].detach_()
			loss_im = loss_im1 + loss_im2

			loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
			total_loss = total_loss + (loss_im + 10.0 * loss_smooth) * self._weights[ii]
			
			loss_sf_2d = loss_sf_2d + loss_im 
			loss_sf_sm = loss_sf_sm + loss_smooth

		loss_dict = {}
		loss_dict["ofd2"] = loss_sf_2d
		loss_dict["ofs2"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		return loss_dict

class Eval_Flow_Only(nn.Module):
	def __init__(self):
		super(Eval_Flow_Only, self).__init__()
	

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		im_l1 = target_dict['input_l1']
		batch_size, _, _, _ = im_l1.size()

		gt_flow = target_dict['target_flow']
		gt_flow_mask = target_dict['target_flow_mask']

		## Flow EPE
		out_flow = self.upsample_flow_as(output_dict['flow_f'][0], gt_flow)
		valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask.float()
		loss_dict["epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
		
		flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
		outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
		loss_dict["f1"] = (outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68

		output_dict["out_flow_pp"] = out_flow

		return loss_dict


class Loss_Disp_Only(nn.Module):
	def __init__(self, args):
		super(Loss_Disp_Only, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1


	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Image loss: 
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['disp_l1'])):
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		## SceneFlow Loss
		batch_size = target_dict['input_l1'].size(0)
		loss_dp_sum = 0

		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(disp_l1.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l2)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l2)

			## Depth Loss
			loss_disp_l1, _ = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, _ = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

		total_loss = loss_dp_sum

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Eval_Disp_Only(nn.Module):
	def __init__(self):
		super(Eval_Disp_Only, self).__init__()


	def forward(self, output_dict, target_dict):
		

		loss_dict = {}

		## Depth Eval
		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1)
		intrinsics = target_dict['input_k_l1']

		out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
		out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
		out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
		gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l1, gt_depth_pp, out_depth_l1)

		output_dict["out_disp_l_pp"] = out_disp_l1
		output_dict["out_depth_l_pp"] = out_depth_l1

		loss_dict["d1"] = output_dict_displ['otl']

		loss_dict["ab"] = output_dict_displ['abs_rel']
		loss_dict["sq"] = output_dict_displ['sq_rel']
		loss_dict["rms"] = output_dict_displ['rms']
		loss_dict["lrms"] = output_dict_displ['log_rms']
		loss_dict["a1"] = output_dict_displ['a1']
		loss_dict["a2"] = output_dict_displ['a2']
		loss_dict["a3"] = output_dict_displ['a3']


		return loss_dict


###############################################
## MonoDepth Experiment
###############################################

class Basis_MonoDepthLoss(nn.Module):
	def __init__(self):
		super(Basis_MonoDepthLoss, self).__init__()
		self.ssim_w = 0.85
		self.disp_gradient_w = 0.1
		self.lr_w = 1.0
		self.n = 4

	def scale_pyramid(self, img_input, depths):
		scaled_imgs = []
		for _, depth in enumerate(depths):
			scaled_imgs.append(interpolate2d_as(img_input, depth))
		return scaled_imgs

	def gradient_x(self, img):
		# Pad input to keep output size consistent
		img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
		gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
		return gx

	def gradient_y(self, img):
		# Pad input to keep output size consistent
		img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
		gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
		return gy

	def apply_disparity(self, img, disp):
		batch_size, _, height, width = img.size()

		# Original coordinates of pixels
		x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
		y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

		# Apply shift in X direction
		x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
		flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
		# In grid_sample coordinates are assumed to be between -1 and 1
		output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

		return output

	def generate_image_left(self, img, disp):
		return self.apply_disparity(img, -disp)

	def generate_image_right(self, img, disp):
		return self.apply_disparity(img, disp)

	def SSIM(self, x, y):
		C1 = 0.01 ** 2
		C2 = 0.03 ** 2

		mu_x = nn.AvgPool2d(3, 1)(x)
		mu_y = nn.AvgPool2d(3, 1)(y)
		mu_x_mu_y = mu_x * mu_y
		mu_x_sq = mu_x.pow(2)
		mu_y_sq = mu_y.pow(2)

		sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
		sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
		sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

		SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
		SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
		SSIM = SSIM_n / SSIM_d

		SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

		return tf.pad(SSIM_img, pad=(1,1,1,1), mode='constant', value=0)

	def disp_smoothness(self, disp, pyramid):
		disp_gradients_x = [self.gradient_x(d) for d in disp]
		disp_gradients_y = [self.gradient_y(d) for d in disp]

		image_gradients_x = [self.gradient_x(img) for img in pyramid]
		image_gradients_y = [self.gradient_y(img) for img in pyramid]

		weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
		weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

		smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
		smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

		return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

	def forward(self, disp_l, disp_r, img_l, img_r):

		self.n = len(disp_l)

		## Image pyramid
		img_l_pyramid = self.scale_pyramid(img_l, disp_l)
		img_r_pyramid = self.scale_pyramid(img_r, disp_r)

		## Disocc map
		right_occ = [_adaptive_disocc_detection_disp(-disp_l[i]) for i in range(self.n)]
		left_occ  = [_adaptive_disocc_detection_disp(disp_r[i]) for i in range(self.n)]


		## Image reconstruction loss
		left_est = [self.generate_image_left(img_r_pyramid[i], disp_l[i]) for i in range(self.n)]
		right_est = [self.generate_image_right(img_l_pyramid[i], disp_r[i]) for i in range(self.n)]

		# L1
		l1_left = [torch.mean((torch.abs(left_est[i] - img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
		l1_right = [torch.mean((torch.abs(right_est[i] - img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

		# SSIM
		ssim_left = [torch.mean((self.SSIM(left_est[i], img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
		ssim_right = [torch.mean((self.SSIM(right_est[i], img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

		image_loss_left = [self.ssim_w * ssim_left[i] + (1 - self.ssim_w) * l1_left[i] for i in range(self.n)]
		image_loss_right = [self.ssim_w * ssim_right[i] + (1 - self.ssim_w) * l1_right[i] for i in range(self.n)]
		image_loss = sum(image_loss_left + image_loss_right)


		## L-R Consistency loss
		right_left_disp = [self.generate_image_left(disp_r[i], disp_l[i]) for i in range(self.n)]
		left_right_disp = [self.generate_image_right(disp_l[i], disp_r[i]) for i in range(self.n)]

		lr_left_loss = [torch.mean((torch.abs(right_left_disp[i] - disp_l[i]))[left_occ[i]]) for i in range(self.n)]
		lr_right_loss = [torch.mean((torch.abs(left_right_disp[i] - disp_r[i]))[right_occ[i]]) for i in range(self.n)]
		lr_loss = sum(lr_left_loss + lr_right_loss)


		## Disparities smoothness
		disp_left_smoothness = self.disp_smoothness(disp_l, img_l_pyramid)
		disp_right_smoothness = self.disp_smoothness(disp_r, img_r_pyramid)

		disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
		disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
		disp_gradient_loss = sum(disp_left_loss + disp_right_loss)


		## Loss sum
		loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss

		return loss

class Loss_MonoDepth(nn.Module):
	def __init__(self):

		super(Loss_MonoDepth, self).__init__()
		self._depth_loss = Basis_MonoDepthLoss()

	def forward(self, output_dict, target_dict):

		loss_dict = {}
		depth_loss = self._depth_loss(output_dict['disp_l1'], output_dict['disp_r1'], target_dict['input_l1'], target_dict['input_r1'])
		loss_dict['total_loss'] = depth_loss

		return loss_dict

class Eval_MonoDepth(nn.Module):
	def __init__(self):
		super(Eval_MonoDepth, self).__init__()

	def forward(self, output_dict, target_dict):
		
		loss_dict = {}

		## Depth Eval
		gt_disp = target_dict['target_disp']
		gt_disp_mask = (target_dict['target_disp_mask']==1)
		intrinsics = target_dict['input_k_l1_orig']

		out_disp_l_pp = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
		out_depth_l_pp = _disp2depth_kitti_K(out_disp_l_pp, intrinsics[:, 0, 0])
		out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
		gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

		output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

		output_dict["out_disp_l_pp"] = out_disp_l_pp
		output_dict["out_depth_l_pp"] = out_depth_l_pp
		loss_dict["ab_r"] = output_dict_displ['abs_rel']
		loss_dict["sq_r"] = output_dict_displ['sq_rel']

		return loss_dict

###############################################


###############################################
#
#   Loss with only sceneflow self-supervised
#
###############################################

class Loss_SceneFlow_Depth_Sup(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_Depth_Sup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def _depth2disp_kitti_K(self, depth, k_value):
		disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

		return disp


	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		gt_disp_l1 = target_dict['disp_l1']
		gt_disp_l2 = target_dict['disp_l2']
		gt_disp_r1 = target_dict['disp_r1']
		gt_disp_r2 = target_dict['disp_r2']

		gt_disp_l1_mask = target_dict['disp_l1_mask']
		gt_disp_l2_mask = target_dict['disp_l2_mask']
		gt_disp_r1_mask = target_dict['disp_r1_mask']
		gt_disp_r2_mask = target_dict['disp_r2_mask']

		b, _, h_dp, w_dp = gt_disp_l1.size()

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			gt_disp_l1 = gt_disp_l1.to(disp_l1.device)
			gt_disp_l2 = gt_disp_l2.to(disp_l1.device)
			## Disp Loss
			_, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			_, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)


			disp_l1_up = interpolate2d_as(disp_l1, gt_disp_l1, mode="bilinear") * w_dp
			disp_l2_up = interpolate2d_as(disp_l2, gt_disp_l2, mode="bilinear") * w_dp
			# disp_r1_up = interpolate2d_as(disp_r1, gt_disp_r1, mode="bilinear") * w_dp
			# disp_r2_up = interpolate2d_as(disp_r2, gt_disp_r2, mode="bilinear") * w_dp
			#print("Device for gt_disp_l1:",gt_disp_l1.device)
			#print("Device for disp_l1_up:",disp_l1_up.device)

			gt_disp_l1 = self._depth2disp_kitti_K(gt_disp_l1, target_dict['input_k_l1_aug'][:, 0, 0])
			gt_disp_l2 = self._depth2disp_kitti_K(gt_disp_l2, target_dict['input_k_l2_aug'][:, 0, 0])

			valid_abs_rel_l1 = torch.abs(gt_disp_l1.to(disp_l1.device) - disp_l1_up) * gt_disp_l1_mask.to(disp_l1.device)
			valid_abs_rel_l1[gt_disp_l1_mask == 0].detach_()
			disp_l1_loss = valid_abs_rel_l1[gt_disp_l1_mask != 0].mean()

			valid_abs_rel_l2 = torch.abs(gt_disp_l2.to(disp_l1.device) - disp_l2_up) * gt_disp_l2_mask.to(disp_l1.device)
			valid_abs_rel_l2[gt_disp_l2_mask == 0].detach_()
			disp_l2_loss = valid_abs_rel_l2[gt_disp_l2_mask != 0].mean()

			# valid_abs_rel_r1 = torch.abs(gt_disp_r1 - disp_r1_up) * gt_disp_r1_mask
			# valid_abs_rel_r1[gt_disp_r1_mask == 0].detach_()
			# disp_r1_loss = valid_abs_rel_r1[gt_disp_r1_mask != 0].mean()

			# valid_abs_rel_r2 = torch.abs(gt_disp_r2 - disp_r2_up) * gt_disp_r2_mask
			# valid_abs_rel_r2[gt_disp_r2_mask == 0].detach_()
			# disp_r2_loss = valid_abs_rel_l1[gt_disp_r2_mask != 0].mean()
			


			loss_dp_sum = loss_dp_sum + (disp_l1_loss + disp_l2_loss) * self._weights[ii]



			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

######################################################################
#
#  Feature Metric Loss
#
######################################################################
class Loss_SceneFlow_SelfSup_FeatMetric(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_FeatMetric, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def robust_l1(self, pred, target):
		eps = 1e-3
		return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def depth_loss_left_features(self, disp_l, disp_r, features_l, features_r):

		features_r_warp = _generate_feat_left(features_r, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		feat_diff = self.robust_l1(features_r_warp, features_l).mean(dim=1, keepdim=True)        
		loss_feat = (feat_diff[left_occ]).mean()
		feat_diff[~left_occ].detach_()

		return loss_feat


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_fm_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		feats_r1 = output_dict['output_dict_r']['x1_feats']
		feats_r2 = output_dict['output_dict_r']['x2_feats']
		feats_l1 = output_dict['x1_feats']
		feats_l2 = output_dict['x2_feats']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_fm_l1 = self.depth_loss_left_features(disp_l1, disp_r1, feats_l1[ii], feats_r1[ii])
			loss_fm_l2 = self.depth_loss_left_features(disp_l2, disp_r2, feats_l2[ii], feats_r2[ii])
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
			loss_fm_sum = loss_fm_sum + (loss_fm_l1 + loss_fm_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		fm_loss = loss_fm_sum.detach()
		#print(f_loss,d_loss,fm_loss)
		max_val = torch.max(torch.max(f_loss, d_loss),fm_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss
		fm_weight = max_val / fm_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight + loss_fm_sum * fm_weight

		loss_dict = {}
		loss_dict["fm"] = loss_fm_sum
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


######################################################################
#
#  Feature Metric Loss + Feature_Regularization Loss
#
######################################################################
class Loss_SceneFlow_SelfSup_FeatMetReg(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SelfSup_FeatMetReg, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self.dis = 1e-3
		self.cvt = 1e-3

	def robust_l1(self, pred, target):
		eps = 1e-3
		return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def depth_loss_left_features(self, disp_l, disp_r, features_l, features_r):

		features_r_warp = _generate_feat_left(features_r, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		feat_diff = self.robust_l1(features_r_warp, features_l).mean(dim=1, keepdim=True)        
		loss_feat = (feat_diff[left_occ]).mean()
		feat_diff[~left_occ].detach_()

		return loss_feat


	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def gradient(self, D):
		D_dy = D[:, :, 1:] - D[:, :, :-1]
		D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
		return D_dx, D_dy

	def feature_regularization_loss(self, feature, img):
		#b, _, h, w = feature.size()
		#img = F.interpolate(img, (h, w), mode='area')

		feature_dx, feature_dy = self.gradient(feature)
		img_dx, img_dy = self.gradient(img)

		feature_dxx, feature_dxy = self.gradient(feature_dx)
		feature_dyx, feature_dyy = self.gradient(feature_dy)

		img_dxx, img_dxy = self.gradient(img_dx)
		img_dyx, img_dyy = self.gradient(img_dy)

		smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
				  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

		smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
				  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
				  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
				  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

		return -self.dis * smooth1+ self.cvt * smooth2

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_fm_sum = 0
		loss_reg_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		feats_r1 = output_dict['output_dict_r']['x1_feats']
		feats_r2 = output_dict['output_dict_r']['x2_feats']
		feats_l1 = output_dict['x1_feats']
		feats_l2 = output_dict['x2_feats']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_fm_l1 = self.depth_loss_left_features(disp_l1, disp_r1, feats_l1[ii], feats_r1[ii])
			loss_fm_l2 = self.depth_loss_left_features(disp_l2, disp_r2, feats_l2[ii], feats_r2[ii])
			loss_reg_l1 = self.feature_regularization_loss(feats_l1[ii],img_l1_aug)
			loss_reg_l2 = self.feature_regularization_loss(feats_l2[ii],img_l2_aug)


			loss_reg_sum = loss_reg_sum + (loss_reg_l1 + loss_reg_l2) * self._weights[ii]
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
			loss_fm_sum = loss_fm_sum + (loss_fm_l1 + loss_fm_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		fm_loss = loss_fm_sum.detach()
		#print(f_loss,d_loss,fm_loss)
		max_val = torch.max(torch.max(f_loss, d_loss),fm_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss
		fm_weight = max_val / fm_loss

		total_loss = loss_sf_sum * f_weight + (loss_dp_sum + loss_reg_sum) * d_weight + loss_fm_sum * fm_weight

		loss_dict = {}
		loss_dict["reg"] = loss_reg_sum
		loss_dict["fm"] = loss_fm_sum
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_SceneFlow_SemiDepthSup(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_SemiDepthSup, self).__init__()        

		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._unsup_loss = Loss_SceneFlow_SelfSup(args)

	def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = sf_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def _depth2disp_kitti_K(self, depth, k_value):
		disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

		return disp

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		unsup_loss_dict = self._unsup_loss(output_dict, target_dict)
		unsup_loss = unsup_loss_dict['total_loss']

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0

		## Ground Truth
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		gt_disp_l1 = target_dict['disp_l1']
		gt_disp_l2 = target_dict['disp_l2']
		gt_disp_r1 = target_dict['disp_r1']
		gt_disp_r2 = target_dict['disp_r2']

		gt_disp_l1_mask = target_dict['disp_l1_mask']
		gt_disp_l2_mask = target_dict['disp_l2_mask']
		gt_disp_r1_mask = target_dict['disp_r1_mask']
		gt_disp_r2_mask = target_dict['disp_r2_mask']

		b, _, h_dp, w_dp = gt_disp_l1.size()

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			gt_disp_l1 = gt_disp_l1.to(disp_l1.device)
			gt_disp_l2 = gt_disp_l2.to(disp_l1.device)
			## Disp Loss
			#disp_occ_l1 = _adaptive_disocc_detection_disp(disp_r1).detach()
			#disp_occ_l2 = _adaptive_disocc_detection_disp(disp_r2).detach()


			disp_l1_up = interpolate2d_as(disp_l1, gt_disp_l1, mode="bilinear") * w_dp
			disp_l2_up = interpolate2d_as(disp_l2, gt_disp_l2, mode="bilinear") * w_dp

			gt_disp_l1 = self._depth2disp_kitti_K(gt_disp_l1, target_dict['input_k_l1_aug'][:, 0, 0])
			gt_disp_l2 = self._depth2disp_kitti_K(gt_disp_l2, target_dict['input_k_l2_aug'][:, 0, 0])

			valid_abs_rel_l1 = torch.abs(gt_disp_l1.to(disp_l1.device) - disp_l1_up) * gt_disp_l1_mask.to(disp_l1.device)
			valid_abs_rel_l1[gt_disp_l1_mask == 0].detach_()
			disp_l1_loss = valid_abs_rel_l1[gt_disp_l1_mask != 0].mean()

			print(gt_disp_l1_mask.sum())

			valid_abs_rel_l2 = torch.abs(gt_disp_l2.to(disp_l1.device) - disp_l2_up) * gt_disp_l2_mask.to(disp_l1.device)
			valid_abs_rel_l2[gt_disp_l2_mask == 0].detach_()
			disp_l2_loss = valid_abs_rel_l2[gt_disp_l2_mask != 0].mean()	

			loss_dp_sum = loss_dp_sum + (disp_l1_loss + disp_l2_loss) * self._weights[ii]
			print(ii, loss_dp_sum)



			## Sceneflow Loss           
			#loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
			#																disp_l1, disp_l2,
		#																	disp_occ_l1, disp_occ_l2,
		#																	k_l1_aug, k_l2_aug,
		#																	img_l1_aug, img_l2_aug, 
		#																	aug_size, ii)

			#loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			#loss_sf_2d = loss_sf_2d + loss_im            
			#loss_sf_3d = loss_sf_3d + loss_pts
			#loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		u_loss = unsup_loss.detach()
		d_loss = loss_dp_sum.detach()
		#sf_loss = loss_sf_sum.detach()

		print("\ndisp loss, unsup loss", d_loss, u_loss)

		max_val = torch.max(d_loss, u_loss)

		u_weight = max_val / u_loss
		d_weight = max_val / d_loss 
		#f_weight = max_val / sf_loss 
		weight = torch.max(u_weight,d_weight)

		total_loss = unsup_loss + loss_dp_sum / weight 
		loss_dict["unsup"] = unsup_loss
		loss_dict["dp"] = loss_dp_sum
		#loss_dict["sf"] = loss_sf_sum
		#loss_dict["s2"] = loss_sf_2d
		#loss_dict["s3"] = loss_sf_3d
		loss_dict["total_loss"] = total_loss

		return loss_dict



###############################################
#
#   Loss with only depth self-supervised
#
###############################################

class Loss_SceneFlow_Sf_Sup(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_Sf_Sup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img1(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def depth_loss_left_img2(self, disp_l, disp_r, img_l_aug, img_r_aug, ii, mask):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = (_adaptive_disocc_detection_disp(disp_r).float()*mask).detach()
		left_occ = left_occ == 1

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def _depth2disp_kitti_K(self, depth, k_value):
		disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

		return disp


	def forward(self, output_dict, target_dict):

		loss_dict = {}
		#print(output_dict.keys())

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		sf_fr_dict = output_dict['output_dict_r']['flow_f']
		sf_br_dict = output_dict['output_dict_r']['flow_b']

		gt_sf_l = target_dict['sf_l'].to(disp_r1_dict[0].device)
		gt_sf_r = target_dict['sf_r'].to(disp_r1_dict[0].device)
		#gt_sf_bl = target_dict['sf_bl'].to(disp_r1_dict[0].device)
		#gt_sf_br = target_dict['sf_br'].to(disp_r1_dict[0].device)
		gt_sf_l_mask = target_dict['valid_sf_l'].to(disp_r1_dict[0].device)
		gt_sf_r_mask = target_dict['valid_sf_r'].to(disp_r1_dict[0].device)

		#gt_im_l_mask = target_dict['valid_pixels_l'].to(disp_r1_dict[0].device)
		#gt_im_r_mask = target_dict['valid_pixels_r'].to(disp_r1_dict[0].device)

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f_pp'], output_dict['flow_b_pp'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_b.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_f)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_f)

			im_mask = interpolate2d_as(gt_sf_l_mask, disp_r1).to(disp_l1.device).float()

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img1(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img2(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii, im_mask)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			sf_f_up = interpolate2d_as(sf_f, gt_sf_l)
			#sf_b_up = interpolate2d_as(sf_b, gt_sf_l)
			#gt_sf_r_down = interpolate2d_as(gt_sf_r, sf_f_r)
			disp_occ_l1 = interpolate2d_as(disp_occ_l1.float(), gt_sf_l_mask)
			disp_occ_l2 = interpolate2d_as(disp_occ_l2.float(), gt_sf_l_mask)
			#sf_r_mask_down = interpolate2d_as(gt_sf_r_mask, sf_f_r)
			sf_f_mask = gt_sf_l_mask * disp_occ_l1.float()
			#sf_b_mask = gt_im_l_mask * disp_occ_l2.float()
			#print("input shape:", sf_f_up.shape)

			valid_epe_f = _elementwise_robust_epe_char(sf_f_up, gt_sf_l) * sf_f_mask
			valid_epe_f[sf_f_mask == 0].detach_()
			flow_f_loss = valid_epe_f[sf_f_mask != 0].mean()

			#valid_epe_b = _elementwise_robust_epe_char(sf_b_up, gt_sf_bl) * sf_b_mask
			#valid_epe_b[sf_b_mask == 0].detach_()
			#flow_b_loss = valid_epe_b[sf_b_mask != 0].mean()
			#valid_epe_r = _elementwise_robust_epe_char(out_flow, gt_flow) * sf_r_mask_down * disp_occ_l2

			loss_sf_sum = loss_sf_sum + flow_f_loss * self._weights[ii]            

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_SceneFlow_Flow_Sup(nn.Module):
	def __init__(self, args):
		super(Loss_SceneFlow_Flow_Sup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img1(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def depth_loss_left_img2(self, disp_l, disp_r, img_l_aug, img_r_aug, ii, mask):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = (_adaptive_disocc_detection_disp(disp_r).float()*mask).detach()
		left_occ = left_occ == 1

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def _depth2disp_kitti_K(self, depth, k_value):
		disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

		return disp


	def forward(self, output_dict, target_dict):

		loss_dict = {}
		#print(output_dict.keys())

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		sf_fr_dict = output_dict['output_dict_r']['flow_f']
		sf_br_dict = output_dict['output_dict_r']['flow_b']

		gt_sf_l = target_dict['sf_l'].to(disp_r1_dict[0].device)
		gt_sf_r = target_dict['sf_r'].to(disp_r1_dict[0].device)
		#gt_sf_bl = target_dict['sf_bl'].to(disp_r1_dict[0].device)
		#gt_sf_br = target_dict['sf_br'].to(disp_r1_dict[0].device)
		gt_sf_l_mask = target_dict['valid_sf_l'].to(disp_r1_dict[0].device)
		gt_sf_r_mask = target_dict['valid_sf_r'].to(disp_r1_dict[0].device)

		#gt_im_l_mask = target_dict['valid_pixels_l'].to(disp_r1_dict[0].device)
		#gt_im_r_mask = target_dict['valid_pixels_r'].to(disp_r1_dict[0].device)

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f_pp'], output_dict['flow_b_pp'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_b.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_f)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_f)

			im_mask = interpolate2d_as(gt_sf_l_mask, disp_r1).to(disp_l1.device).float()

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img1(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img2(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii, im_mask)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			sf_f_up = interpolate2d_as(sf_f, gt_sf_l)
			#sf_b_up = interpolate2d_as(sf_b, gt_sf_l)
			#gt_sf_r_down = interpolate2d_as(gt_sf_r, sf_f_r)
			disp_occ_l1 = interpolate2d_as(disp_occ_l1.float(), gt_sf_l_mask)
			disp_occ_l2 = interpolate2d_as(disp_occ_l2.float(), gt_sf_l_mask)
			#sf_r_mask_down = interpolate2d_as(gt_sf_r_mask, sf_f_r)
			sf_f_mask = gt_sf_l_mask * disp_occ_l1.float()
			#sf_b_mask = gt_im_l_mask * disp_occ_l2.float()
			#print("input shape:", sf_f_up.shape)

			valid_epe_f = _elementwise_epe(sf_f_up, gt_sf_l / 20.0) * sf_f_mask
			valid_epe_f[sf_f_mask == 0].detach_()
			flow_f_loss = valid_epe_f[sf_f_mask != 0].mean()

			#valid_epe_b = _elementwise_robust_epe_char(sf_b_up, gt_sf_bl) * sf_b_mask
			#valid_epe_b[sf_b_mask == 0].detach_()
			#flow_b_loss = valid_epe_b[sf_b_mask != 0].mean()
			#valid_epe_r = _elementwise_robust_epe_char(out_flow, gt_flow) * sf_r_mask_down * disp_occ_l2

			loss_sf_sum = loss_sf_sum + flow_f_loss * self._weights[ii]            

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


#################################################################################
#
# Loss for Self-Sup with Network as flow + disp + expansion as multiplier
#
#################################################################################
class Loss_SelfSup_SF_Disp_Exp(nn.Module):
	def __init__(self, args):
		super(Loss_SelfSup_SF_Disp_Exp, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, flow_f, flow_b, exp_f, exp_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		sf_f = torch.cat([flow_f,disp_l1*(exp_f-1)],dim=1)
		sf_b = torch.cat([flow_b,disp_l2*(exp_b-1)],dim=1)
		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		#flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		#flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		#exp_r1_dict = output_dict['output_dict_r']['exp_l1']
		#exp_r2_dict = output_dict['output_dict_r']['exp_l2']

		for ii, (flow_f, flow_b, disp_l1, disp_l2, disp_r1, disp_r2, exp_f, exp_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict, output_dict['exp_f'], output_dict['exp_b'])):

			assert(flow_f.size()[2:4] == flow_b.size()[2:4])
			assert(flow_f.size()[2:4] == disp_l1.size()[2:4])
			assert(flow_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], flow_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], flow_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], flow_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], flow_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(flow_f, flow_b,
																			exp_f, exp_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

#################################################################################
#
# Loss for Self-Sup with Network as flow + disp + expansion as offset
#
#################################################################################
class Loss_SelfSup_SF_Disp_Exp_Plus(nn.Module):
	def __init__(self, args):
		super(Loss_SelfSup_SF_Disp_Exp_Plus, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, flow_f, flow_b, exp_f, exp_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp
		exp_f = exp_f * w_dp
		exp_b = exp_b * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp         

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		sf_f = torch.cat([flow_f,exp_f],dim=1)
		sf_b = torch.cat([flow_b,exp_b],dim=1)
		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		#flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		#flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		## Image reconstruction loss
		img_l2_warp = reconstructImg(coord1, img_l2_aug)
		img_l1_warp = reconstructImg(coord2, img_l1_aug)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		## 3D motion smoothness loss
		loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		#exp_r1_dict = output_dict['output_dict_r']['exp_l1']
		#exp_r2_dict = output_dict['output_dict_r']['exp_l2']

		for ii, (flow_f, flow_b, disp_l1, disp_l2, disp_r1, disp_r2, exp_f, exp_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict, output_dict['exp_f'], output_dict['exp_b'])):

			assert(flow_f.size()[2:4] == flow_b.size()[2:4])
			assert(flow_f.size()[2:4] == disp_l1.size()[2:4])
			assert(flow_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], flow_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], flow_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], flow_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], flow_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(flow_f, flow_b,
																			exp_f, exp_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict



class Loss_FlowDisp_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10.0 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug)

			loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im
			loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_FlowDisp_SelfSup_Census(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_Census, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 1
		#print(kernel_size)

		## Photometric loss
		img_diff, mask = census_loss(img_l_aug, img_r_warp, kernel_size)
		img_diff = img_diff.mean(dim=1, keepdim=True)
		left_occ = left_occ * mask
		#img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)
		#print(torch.isnan(loss_img), torch.isnan(loss_smooth))

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, ii):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 3
		#print(kernel_size)

		img_diff1, mask_f = census_loss(img_l1, img_l2_warp, kernel_size)
		img_diff2, mask_b = census_loss(img_l2, img_l1_warp, kernel_size)
		#img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		occ_map_f = occ_map_f * mask_f
		occ_map_b = occ_map_b * mask_b
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'][:4], output_dict['flow_b'][:4], output_dict['disp_l1'][:4], output_dict['disp_l2'][:4], disp_r1_dict[:4], disp_r2_dict[:4])):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

			## Sceneflow Loss           
			loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug, ii)
			#print(loss_im, loss_smooth)

			loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im
			loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		#print(loss_sf_sum, loss_dp_sum)

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 1
		#print(kernel_size)

		## Photometric loss
		img_diff= (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)       
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)
		#print(torch.isnan(loss_img), torch.isnan(loss_smooth))

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, ii):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 3
		#print(kernel_size)

		img_diff1, mask_f = census_loss(img_l1, img_l2_warp, kernel_size)
		img_diff2, mask_b = census_loss(img_l2, img_l1_warp, kernel_size)
		#img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		occ_map_f = occ_map_f * mask_f
		occ_map_b = occ_map_b * mask_b
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'][:4], output_dict['flow_b'][:4], output_dict['disp_l1'][:4], output_dict['disp_l2'][:4], disp_r1_dict[:4], disp_r2_dict[:4])):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

			## Sceneflow Loss           
			loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug, ii)
			#print(loss_im, loss_smooth)

			loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im
			loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		#print(loss_sf_sum, loss_dp_sum)

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_Top(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_Top, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 1
		#print(kernel_size)

		## Photometric loss
		img_diff= (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)       
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)
		#print(torch.isnan(loss_img), torch.isnan(loss_smooth))

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, ii):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 3
		#print(kernel_size)

		img_diff1, mask_f = census_loss(img_l1, img_l2_warp, kernel_size)
		img_diff2, mask_b = census_loss(img_l2, img_l1_warp, kernel_size)
		#img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		occ_map_f = occ_map_f * mask_f
		occ_map_b = occ_map_b * mask_b
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		sf_f = output_dict['flow_f'][0]
		sf_b = output_dict['flow_b'][0]
		disp_l1 = output_dict['disp_l1'][0]
		disp_l2 = output_dict['disp_l2'][0]
		disp_r1 = disp_r1_dict[0]
		disp_r2 = disp_r2_dict[0]

		assert(sf_f.size()[2:4] == sf_b.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
		## For image reconstruction loss
		img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
		img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
		img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
		img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

		## Disp Loss
		loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, 0)
		loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, 0)
		loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2)

		## Sceneflow Loss           
		loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug, 0)
		#print(loss_im, loss_smooth)

		loss_sf_sum = loss_sf_sum + loss_flow            
		loss_sf_2d = loss_sf_2d + loss_im
		loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		#print(loss_sf_sum, loss_dp_sum)

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_v2(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_CensusFlow_SSIM_Disp_v2, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 1
		img_diff1, mask = census_loss(img_l_aug, img_r_warp, kernel_size)
		left_occ = left_occ * mask
		#print(kernel_size)

		## Photometric loss
		img_diff= (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True) + img_diff1 * 0.08      
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)
		#print(torch.isnan(loss_img), torch.isnan(loss_smooth))

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, ii):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 3
		#print(kernel_size)

		img_diff1, mask_f = census_loss(img_l1, img_l2_warp, kernel_size)
		img_diff2, mask_b = census_loss(img_l2, img_l1_warp, kernel_size)
		#img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		occ_map_f = occ_map_f * mask_f
		occ_map_b = occ_map_b * mask_b
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'][:4], output_dict['flow_b'][:4], output_dict['disp_l1'][:4], output_dict['disp_l2'][:4], disp_r1_dict[:4], disp_r2_dict[:4])):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

			## Sceneflow Loss           
			loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug, ii)
			#print(loss_im, loss_smooth)

			loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im
			loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		#print(loss_sf_sum, loss_dp_sum)

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_FlowDisp_SelfSup_Census_Top(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_Census_Top, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 1
		#print(kernel_size)

		## Photometric loss
		img_diff, mask = census_loss(img_l_aug, img_r_warp, kernel_size)
		img_diff = img_diff.mean(dim=1, keepdim=True)
		left_occ = left_occ * mask
		#img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)
		#print(torch.isnan(loss_img), torch.isnan(loss_smooth))

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, ii):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		kernel_size = (ii == 0) * 7 + ((ii == 1) or (ii == 2)) * 5 + (ii == 3) * 3 + (ii == 4) * 3
		#print(kernel_size)

		img_diff1, mask_f = census_loss(img_l1, img_l2_warp, kernel_size)
		img_diff2, mask_b = census_loss(img_l2, img_l1_warp, kernel_size)
		#img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		#img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		occ_map_f = occ_map_f * mask_f
		occ_map_b = occ_map_b * mask_b
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2
		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10 * loss_smooth)
			
		return total_loss, loss_im, loss_smooth

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		sf_f = output_dict['flow_f'][0]
		sf_b = output_dict['flow_b'][0]
		disp_l1 = output_dict['disp_l1'][0]
		disp_l2 = output_dict['disp_l2'][0]
		disp_r1 = disp_r1_dict[0]
		disp_r2 = disp_r2_dict[0]

		assert(sf_f.size()[2:4] == sf_b.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
		## For image reconstruction loss
		img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
		img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
		img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
		img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

		## Disp Loss
		loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, 0)
		loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, 0)
		loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2)

		## Sceneflow Loss           
		loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug, 0)
		#print(loss_im, loss_smooth)

		loss_sf_sum = loss_sf_sum + loss_flow            
		loss_sf_2d = loss_sf_2d + loss_im
		loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		#print(loss_sf_sum, loss_dp_sum)

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_Exp_Sup(nn.Module):
	def __init__(self, args):
		super(Loss_Exp_Sup, self).__init__()
				
		self._args = args
		self.fac = 1 
		self.maxdisp = 256

	def forward(self, output_dict, input_dict):

		loss_dict = {}

		batch_size = input_dict['flow_f'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0

		loss_dc_f = 0
		loss_dc_b = 0
		loss_iexp_f = 0
		loss_iexp_b = 0

		gt_flow_f = input_dict['flow_f'][:,:2,:,:].cuda()
		gt_mask_f = output_dict['mask_f'].float()

		#print("gt_flow shape and device", gt_flow.shape, gt_flow.device)
		#print("gt_flow mask shape:", gt_flow_mask.shape)
		#print("in_range_mask shape:", in_range_mask.shape)
		#print("output flow device:", output_dict['flows_f'][-1].device)

		# evaluate flow
		out_flow_f = interpolate2d_as(output_dict['flows_f'][-1] * 20, gt_flow_f, mode="bilinear")
		#print("out_flow device:", out_flow.device)
		valid_epe = _elementwise_epe(out_flow_f, gt_flow_f) * gt_mask_f
		aepe = valid_epe.mean()

		#print(output_dict['flows_f'][-1].shape)

		loss_dc_f = output_dict['loss_dc_f']
		loss_iexp_f = output_dict['loss_iexp_f']
		if not self._args.finetuning:
			#print("training using loss backward set")
			gt_flow_b = input_dict['flow_b'][:,:2,:,:].cuda()
			out_flow_b = interpolate2d_as(output_dict['flows_b'][-1] * 20, gt_flow_b, mode="bilinear")
			gt_mask_b = output_dict['mask_b'].float()
			loss_dc_b = output_dict['loss_dc_b']
			loss_iexp_b = output_dict['loss_iexp_b']
			iexp_b = loss_iexp_b.detach()
			dc_b = loss_dc_b.detach()
			loss_dict["iexp_b"] = iexp_b
			loss_dict["dc_b"] = dc_b
			aepe = (aepe + (_elementwise_epe(out_flow_b, gt_flow_b) * gt_mask_b).mean()) / 2
		else:
			loss_dc_b = 0
			loss_iexp_b = 0
		

		# loss
		iexp_f = loss_iexp_f.detach()
		dc_f = loss_dc_f.detach()

		total_loss = loss_dc_f + loss_iexp_b + loss_dc_b + loss_iexp_b

		loss_dict = {}
		loss_dict["iexp_f"] = iexp_f
		loss_dict["dc_f"] = dc_f
		loss_dict["aepe"] = aepe
		loss_dict["total_loss"] = total_loss

		return loss_dict




class Loss_PWCDisp_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_PWCDisp_SelfSup, self).__init__()
				
		self._weights = [1.0, 1.0, 1.0, 1.0, 2.0, 4.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['disp_r0'])):
			output_dict['disp_r0'][ii].detach_()
			output_dict['disp_r1'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		#disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		#disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		#print("len disp:", len(output_dict['disp_r0']))

		for ii, (disp_l0, disp_l1, disp_r0, disp_r1) in enumerate(zip(output_dict['disp_l0'], output_dict['disp_l1'], output_dict['disp_r0'], output_dict['disp_r1'])):

			#assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			#assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			#assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l1)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l1)
			#print("disp_size:", disp_l0.shape)

			## Disp Loss
			loss_disp_l0, disp_occ_l0 = self.depth_loss_left_img(disp_l0, disp_r0, img_l1_aug, img_r1_aug, ii)
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l0 + loss_disp_l1) * self._weights[ii]


			## Sceneflow Loss           
			#loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug)

			#loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			#loss_sf_2d = loss_sf_2d + loss_im
			#loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		#f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		#max_val = torch.max(f_loss, d_loss)
		#f_weight = max_val / f_loss
		#d_weight = max_val / d_loss

		total_loss = loss_dp_sum

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		#loss_dict["flow"] = loss_sf_sum
		#loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict)

		return loss_dict


class Loss_PWCDisp_Unfreeze_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_PWCDisp_Unfreeze_SelfSup, self).__init__()
				
		self._weights = [1.0, 1.0, 1.0, 1.0, 2.0, 4.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['disp_r0'])):
			output_dict['disp_r0'][ii].detach_()
			output_dict['disp_r1'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_flow_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		#disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		#disp_r2_dict = output_dict['output_dict_r']['disp_l2']
		#print("len disp:", len(output_dict['disp_r0']))
		#print("len flow:", len(output_dict['flows_l']))
		#print("disp shape:", output_dict['disp_l0'][0].shape, output_dict['disp_l0'][1].shape, output_dict['disp_l0'][2].shape, output_dict['disp_l0'][3].shape, output_dict['disp_l0'][4].shape, output_dict['disp_l0'][5].shape)
		#print("flow shape:", output_dict['flows_l'][0].shape, output_dict['flows_l'][1].shape, output_dict['flows_l'][2].shape, output_dict['flows_l'][3].shape, output_dict['flows_l'][4].shape)

		for ii, (disp_l0, disp_l1, disp_r0, disp_r1, flow_l) in enumerate(zip(output_dict['disp_l0'][1:], output_dict['disp_l1'][1:], output_dict['disp_r0'][1:], output_dict['disp_r1'][1:], output_dict['flows_l'])):

			#assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			#assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			#assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l1)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l1)
			#print("disp_size:", disp_l0.shape)
			img_l1_flow = interpolate2d_as(target_dict["input_l1_aug"], flow_l)
			img_l2_flow = interpolate2d_as(target_dict["input_l2_aug"], flow_l)
			#img_r1_flow = interpolate2d_as(target_dict["input_r1_aug"], flow_l)
			#img_r2_flow = interpolate2d_as(target_dict["input_r2_aug"], flow_l)

			## Disp Loss
			loss_disp_l0, disp_occ_l0 = self.depth_loss_left_img(disp_l0, disp_r0, img_l1_aug, img_r1_aug, ii)
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l0 + loss_disp_l1) * self._weights[ii]

			# flow_loss
			img_l2_warp = self._warping_layer(img_l2_flow, flow_l)
			occ = _adaptive_disocc_detection(flow_l).detach()

			img_diff1 = (_elementwise_l1(img_l1_flow, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_flow, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
			loss_im1 = img_diff1[occ].mean()
			img_diff1[~occ].detach_()

			loss_flow_sum = loss_flow_sum + loss_im1
			## Sceneflow Loss           
			#loss_flow, loss_im, loss_smooth = self.flow_loss(sf_f, sf_b, img_l1_aug, img_l2_aug)

			#loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]            
			#loss_sf_2d = loss_sf_2d + loss_im
			#loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_flow_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_dp_sum * d_weight + loss_flow_sum * f_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_flow_sum
		#loss_dict["im"] = loss_sf_2d
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict)

		return loss_dict


##########################################################################
#
# Self-Sup Loss for MonoFlowExp_ppV1: predict as flow, disp, and expansion
# 									  for expansion, there is no link to flow
#									  expansion is simply multiplier on disp
#
##########################################################################
class Loss_MonoFlowExp_SelfSup_ppV1(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowExp_SelfSup_ppV1, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._flow_sm = 10
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, exp_f, exp_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf(flow_f, disp_l1, exp_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf(flow_b, disp_l2, exp_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		#flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		#flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		#print(occ_map_f.shape)

		mask_f = self.affine(flow_f).unsqueeze(1) * occ_map_f
		mask_b = self.affine(flow_b).unsqueeze(1) * occ_map_b
		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[mask_f].mean()
		loss_im2 = img_diff2[mask_b].mean()
		img_diff1[~mask_f].detach_()
		img_diff2[~mask_b].detach_()
		loss_im = loss_im1 + loss_im2

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_exp_s = ((_smoothness_motion_2nd(exp_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(exp_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = self._flow_sm * loss_flow_s + self._disp_smooth_w * loss_exp_s

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + loss_3d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def affine(self, flow, pw=1):
		b,_,lh,lw=flow.shape
		pref = get_grid_exp(b,lh,lw)[:,0].permute(0,3,1,2).repeat(b,1,1,1).clone()
		ptar = pref + flow
		pw = 1
		pref = tf.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
		ptar = tf.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
		pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
		ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

		prefprefT = pref.matmul(pref.permute(0,2,1))
		ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
		ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

		Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
		Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

		Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
		exp = Avol.sqrt()
		mask = (exp>0.5) & (exp<2) & (Error<0.1)
		mask = mask[:,0]

		exp = exp.clamp(0.5,2)
		exp[Error>0.1]=1
		return mask

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, disp_l1, disp_l2, exp_f, exp_b, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], output_dict['exp_f'], output_dict['exp_b'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(exp_f, exp_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_sm = loss_sf_sm + loss_3d_s

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


##########################################################################
#
# Self-Sup Loss for MonoFlowExp_ppV1: predict as flow, disp, and expansion
# 									  for expansion, there is no link to flow
#									  expansion is simply multiplier on disp
#
##########################################################################
class Loss_MonoExp_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_MonoExp_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.5
		self._sf_3d_sm = 200
		self._flow_sm = 10
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, exp_f, exp_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_exp(flow_f, disp_l1, exp_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_exp(flow_b, disp_l2, exp_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		pred_flow_f, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		pred_flow_b, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		#flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		#flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		loss_im1 = loss_flow_f[occ_map_f].mean()
		loss_im2 = loss_flow_b[occ_map_b].mean()
		loss_flow_f[~occ_map_f].detach_()
		loss_flow_b[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		#loss_exp_s = ((_smoothness_motion_2nd(exp_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(exp_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		#loss_3d_s = self._disp_smooth_w * loss_exp_s

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts
		
		return sceneflow_loss, loss_im, loss_pts

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		sf_f = output_dict['scaled_flow_f']
		sf_b = output_dict['scaled_flow_b']
		exp_f = output_dict['dchange_f']
		exp_b = output_dict['dchange_b']
		mask_f = output_dict['mask_f'].unsqueeze(1)
		mask_b = output_dict['mask_b'].unsqueeze(1)
			
		## For image reconstruction loss
		disp_l1 = interpolate2d_as(output_dict['disp_l1_pp'][0], sf_f)
		disp_l2 = interpolate2d_as(output_dict['disp_l2_pp'][0], sf_b)
		disp_r1 = interpolate2d_as(disp_r1_dict[0], sf_f)
		disp_r2 = interpolate2d_as(disp_r2_dict[0], sf_b)

		assert(sf_f.size()[2:4] == sf_b.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l2.size()[2:4])

		img_l1_aug = target_dict['input_l1_aug']
		img_l2_aug = target_dict['input_l2_aug']
		img_r1_aug = target_dict['input_r1_aug']
		img_r2_aug = target_dict['input_r2_aug']

		## Disp Loss
		loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, 0)
		loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, 0)
		#print(disp_occ_l2.dtype)

		## Sceneflow Loss           
		loss_sceneflow, loss_im, loss_pts = self.sceneflow_loss(exp_f, exp_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1 * mask_f, disp_occ_l2 * mask_b,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, 0)

		loss_dict = {}
		loss_dict["dp"] = loss_disp_l1 + loss_disp_l2
		loss_dict["s_2"] = loss_im
		loss_dict["s_3"] = loss_pts
		loss_dict["total_loss"] = loss_sceneflow

		return loss_dict



class Loss_Joint_MonoExp_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_Joint_MonoExp_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._flow_sm = 10
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, exp_f, exp_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_exp(flow_f, disp_l1, exp_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_exp(flow_b, disp_l2, exp_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		pred_flow_f, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		pred_flow_b, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		#flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		#flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		loss_flo_f = loss_flow_f[occ_map_f].mean()
		loss_flo_b = loss_flow_b[occ_map_b].mean()
		loss_flow_f[~occ_map_f].detach_()
		loss_flow_b[~occ_map_b].detach_()
		loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_exp_s = ((_smoothness_motion_2nd(exp_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(exp_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = self._disp_smooth_w * loss_exp_s

		loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_flo + self._sf_3d_pts * loss_pts + loss_3d_s + 10 * loss_2d_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_2d_s, loss_flo

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		sf_f = output_dict['scaled_flow_f']
		sf_b = output_dict['scaled_flow_b']
		exp_f = output_dict['dchange_f']
		exp_b = output_dict['dchange_b']
		mask_f = output_dict['mask_f'].unsqueeze(1)
		mask_b = output_dict['mask_b'].unsqueeze(1)
			
		## For image reconstruction loss
		disp_l1 = interpolate2d_as(output_dict['disp_l1_pp'][0], sf_f)
		disp_l2 = interpolate2d_as(output_dict['disp_l2_pp'][0], sf_b)
		disp_r1 = interpolate2d_as(disp_r1_dict[0], sf_f)
		disp_r2 = interpolate2d_as(disp_r2_dict[0], sf_b)

		assert(sf_f.size()[2:4] == sf_b.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
		assert(sf_f.size()[2:4] == disp_l2.size()[2:4])

		img_l1_aug = target_dict['input_l1_aug']
		img_l2_aug = target_dict['input_l2_aug']
		img_r1_aug = target_dict['input_r1_aug']
		img_r2_aug = target_dict['input_r2_aug']

		## Disp Loss
		loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, 0)
		loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, 0)
		#print(disp_occ_l2.dtype)

		## Sceneflow Loss           
		loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_2d_s, loss_flo = self.sceneflow_loss(exp_f, exp_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1 * mask_f, disp_occ_l2 * mask_b,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, 0)

		loss_dict = {}
		loss_dict["dp"] = loss_disp_l1 + loss_disp_l2
		loss_dict["s_2"] = loss_im
		loss_dict["s_3"] = loss_pts
		loss_dict["s_3s"] = loss_3d_s
		loss_dict["s_2s"] = loss_2d_s
		loss_dict["flo"] = loss_flo
		loss_dict["total_loss"] = loss_sceneflow

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_MonoFlowDispC_SelfSup_No_Flow_Reg(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDispC_SelfSup_No_Flow_Reg, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_dispC(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_dispC(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		#loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		#loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		#loss_flo_f = loss_flow_f[occ_map_f].mean()
		#loss_flo_b = loss_flow_b[occ_map_b].mean()
		#loss_flow_f[~occ_map_f].detach_()
		#loss_flow_b[~occ_map_b].detach_()
		#loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + loss_3d_s + 10 * loss_flow_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_flow_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_2d_sm = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_2d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_2d_sm = loss_sf_2d_sm + loss_2d_s
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_2s"] = loss_sf_2d_sm
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v2(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v2, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_dispC_v2(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_dispC_v2(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		#loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		#loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		#loss_flo_f = loss_flow_f[occ_map_f].mean()
		#loss_flo_b = loss_flow_b[occ_map_b].mean()
		#loss_flow_f[~occ_map_f].detach_()
		#loss_flow_b[~occ_map_b].detach_()
		#loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + loss_3d_s + 10 * loss_flow_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_flow_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_2d_sm = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_2d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_2d_sm = loss_sf_2d_sm + loss_2d_s
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_2s"] = loss_sf_2d_sm
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v3(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v3, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.1
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()
		self._args = args
		self._3d_weight = 0

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii, detach_flag):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_dispC_v3(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_dispC_v3(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		#loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		#loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		#loss_flo_f = loss_flow_f[occ_map_f].mean()
		#loss_flo_b = loss_flow_b[occ_map_b].mean()
		#loss_flow_f[~occ_map_f].detach_()
		#loss_flow_b[~occ_map_b].detach_()
		#loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = self._3d_weight * (loss_pts1 + loss_pts2)

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f / 4 , img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b / 4, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = self._3d_weight * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		if detach_flag == False:
			sceneflow_loss = loss_im + loss_pts + loss_3d_s + 10 * loss_flow_s
		else:
			sceneflow_loss = loss_im + 10 * loss_flow_s
			loss_pts.detach()
			loss_flow_s.detach()


		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_flow_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_2d_sm = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		#print(target_dict['input_l1'].shape)
		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		with open('%s/iter_counts.txt'%(self._args.save), 'r') as f:
			cur_iter = int(f.read())
		f.close()

		#print(cur_iter)
		interval = 5 * 25800 / (batch_size)
		if (cur_iter >= self._args.start) & (cur_iter <= (self._args.start + interval)):
			self._3d_weight = (cur_iter - self._args.start) * self._sf_3d_pts / (interval)
			detach_flag = False
		elif (cur_iter < self._args.start):
			detach_flag = True
			self._3d_weight = 0
		else:
			self._3d_weight = self._sf_3d_pts
			detach_flag = False

		#print(self._3d_weight)

		#print(self._3d_weight)


		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_2d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii, detach_flag)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_2d_sm = loss_sf_2d_sm + loss_2d_s
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_2s"] = loss_sf_2d_sm
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict

class Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v4(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDispC_SelfSup_No_Flow_Reg_v4, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.1
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_dispC_v3(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_dispC_v3(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
		occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
		occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		#loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		#loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		#loss_flo_f = loss_flow_f[occ_map_f].mean()
		#loss_flo_b = loss_flow_b[occ_map_b].mean()
		#loss_flow_f[~occ_map_f].detach_()
		#loss_flow_b[~occ_map_b].detach_()
		#loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		loss_flow_s = (_smoothness_motion_2nd(flow_f / 20, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(flow_b / 20, img_l2_aug, beta=10).mean()) / (2 ** ii)
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + loss_3d_s + 10 * loss_flow_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_flow_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_2d_sm = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_2d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_2d_sm = loss_sf_2d_sm + loss_2d_s
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_2s"] = loss_sf_2d_sm
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])

		return loss_dict


class Loss_FlowDisp_SelfSup_TS(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_TS, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, occ_teacher, disp_teacher, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()
		#print(left_occ.shape)
		#print(occ_teacher.shape)

		## student supervision
		mask = torch.clamp((1 - left_occ.float()) - (1 - occ_teacher.float()), 0 , 1)
		disp_ts = _elementwise_robust_epe_char(disp_l, disp_teacher)
		disp_ts_loss = torch.sum(disp_ts * mask) / (torch.sum(mask) + 1e-6)
		#disp_ts[~mask.bool()].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, disp_ts_loss, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, teacher_flow_f, teacher_flow_b, occ_teacher_f, occ_teacher_b):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()

		mask_f = torch.clamp((1 - occ_map_f.float()) - (1 - occ_teacher_f.float()), 0 , 1)
		flow_f_ts = _elementwise_robust_epe_char(sf_f, teacher_flow_f)
		flow_f_ts_loss = torch.sum(flow_f_ts * mask_f) / (torch.sum(mask_f) + 1e-6)
		#flow_f_ts[~mask_f.bool()].detach_()
		
		mask_b = torch.clamp((1 - occ_map_b.float()) - (1 - occ_teacher_b.float()), 0 , 1)
		flow_b_ts = _elementwise_robust_epe_char(sf_b, teacher_flow_b)
		flow_b_ts_loss = torch.sum(flow_b_ts * mask_b) / (torch.sum(mask_b) + 1e-6)
		#flow_b_ts[~mask_b.bool()].detach_()

		loss_im = loss_im1 + loss_im2 
		loss_ts_flow = flow_f_ts_loss + flow_b_ts_loss

		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10.0 * loss_smooth) + loss_ts_flow
			
		return total_loss, loss_im, loss_smooth, loss_ts_flow

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		teacher_dict = output_dict['teacher_dict']
		student_dict = output_dict['student_dict']
		crop_info = output_dict['crop_info']
		str_x = crop_info[0]
		str_y = crop_info[1]
		end_x = crop_info[2]
		end_y = crop_info[3]

		ii = 0
		
		sf_f_student = student_dict['flow_f'][ii]
		sf_b_student = student_dict['flow_b'][ii]
		disp_l1_student = student_dict['disp_l1'][ii]
		disp_l2_student = student_dict['disp_l2'][ii]
		disp_r1_student = student_dict['output_dict_r']['disp_l1'][ii]
		disp_r2_student = student_dict['output_dict_r']['disp_l2'][ii]

		sf_f_teacher = teacher_dict['flow_f_pp'][ii]
		sf_b_teacher = teacher_dict['flow_b_pp'][ii]
		disp_l1_teacher = teacher_dict['disp_l1_pp'][ii]
		disp_l2_teacher = teacher_dict['disp_l2_pp'][ii]
		disp_r1_teacher = teacher_dict['output_dict_r']['disp_l1'][ii]
		disp_r2_teacher = teacher_dict['output_dict_r']['disp_l2'][ii]
			
		## For image reconstruction loss
		img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f_student)
		img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b_student)
		img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f_student)
		img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b_student)

		#print(disp_l1_teacher.requires_grad)

		# teacher occ maskes:
		disp_occ_l1_teacher = _adaptive_disocc_detection_disp(disp_r1_teacher).detach()
		disp_occ_l1_teacher = disp_occ_l1_teacher[:, :, str_y:end_y, str_x:end_x]
		disp_l1_teacher = disp_l1_teacher[:, :, str_y:end_y, str_x:end_x]
		disp_occ_l2_teacher = _adaptive_disocc_detection_disp(disp_r2_teacher).detach()
		disp_occ_l2_teacher = disp_occ_l2_teacher[:, :, str_y:end_y, str_x:end_x]
		disp_l2_teacher = disp_l2_teacher[:, :, str_y:end_y, str_x:end_x]
		flow_occ_f_teacher = _adaptive_disocc_detection(sf_f_teacher).detach()
		flow_occ_f_teacher = flow_occ_f_teacher[:, :, str_y:end_y, str_x:end_x]
		sf_f_teacher = sf_f_teacher[:, :, str_y:end_y, str_x:end_x]
		flow_occ_b_teacher = _adaptive_disocc_detection(sf_b_teacher).detach()
		flow_occ_b_teacher = flow_occ_b_teacher[:, :, str_y:end_y, str_x:end_x]
		sf_b_teacher = sf_b_teacher[:, :, str_y:end_y, str_x:end_x]

		## Disp Loss
		loss_disp_l1, disp_ts_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1_student, disp_r1_student, img_l1_aug, img_r1_aug, disp_occ_l1_teacher, disp_l1_teacher, 0)
		loss_disp_l2, disp_ts_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2_student, disp_r2_student, img_l2_aug, img_r2_aug, disp_occ_l2_teacher, disp_l2_teacher, 0)
		disp_ts = disp_ts_l1 + disp_ts_l2
		loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii] + disp_ts


		## Sceneflow Loss           
		loss_flow, loss_im, loss_smooth, flow_ts = self.flow_loss(sf_f_student, sf_b_student, img_l1_aug, img_l2_aug, sf_f_teacher, sf_b_teacher, flow_occ_f_teacher, flow_occ_b_teacher)



		loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]           
		loss_sf_2d = loss_sf_2d + loss_im
		loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["dp_ts"] = disp_ts
		loss_dict["f_ts"] = flow_ts
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(student_dict['output_dict_r'])

		return loss_dict


class Loss_FlowDisp_SelfSup_TS_OG_size(nn.Module):
	def __init__(self, args):
		super(Loss_FlowDisp_SelfSup_TS_OG_size, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, occ_teacher, disp_teacher, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		## Photometric loss
		img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
		loss_img = (img_diff[left_occ]).mean()
		img_diff[~left_occ].detach_()
		#print(left_occ.shape)
		#print(occ_teacher.shape)

		## student supervision
		mask = torch.clamp((1 - left_occ.float()) - (1 - occ_teacher.float()), 0 , 1)
		disp_ts = _elementwise_robust_epe_char(disp_l, disp_teacher)
		disp_ts_loss = torch.sum(disp_ts * mask) / (torch.sum(mask) + 1e-6)
		#disp_ts[~mask.bool()].detach_()

		## Disparities smoothness
		loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

		return loss_img + self._disp_smooth_w * loss_smooth, disp_ts_loss, left_occ


	def flow_loss(self, sf_f, sf_b, img_l1, img_l2, teacher_flow_f, teacher_flow_b, occ_teacher_f, occ_teacher_b):

		img_l2_warp = self._warping_layer(img_l2, sf_f)
		img_l1_warp = self._warping_layer(img_l1, sf_b)
		occ_map_f = _adaptive_disocc_detection(sf_b).detach()
		occ_map_b = _adaptive_disocc_detection(sf_f).detach()

		img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()

		mask_f = torch.clamp((1 - occ_map_f.float()) - (1 - occ_teacher_f.float()), 0 , 1)
		flow_f_ts = _elementwise_robust_epe_char(sf_f, teacher_flow_f)
		flow_f_ts_loss = torch.sum(flow_f_ts * mask_f) / (torch.sum(mask_f) + 1e-6)
		#flow_f_ts[~mask_f.bool()].detach_()
		
		mask_b = torch.clamp((1 - occ_map_b.float()) - (1 - occ_teacher_b.float()), 0 , 1)
		flow_b_ts = _elementwise_robust_epe_char(sf_b, teacher_flow_b)
		flow_b_ts_loss = torch.sum(flow_b_ts * mask_b) / (torch.sum(mask_b) + 1e-6)
		#flow_b_ts[~mask_b.bool()].detach_()

		loss_im = loss_im1 + loss_im2 
		loss_ts_flow = flow_f_ts_loss + flow_b_ts_loss

		loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
			
		total_loss = (loss_im + 10.0 * loss_smooth) + loss_ts_flow
			
		return total_loss, loss_im, loss_smooth, loss_ts_flow

	def upsample_flow_as(self, flow, output_as):
		size_inputs = flow.size()[2:4]
		size_targets = output_as.size()[2:4]
		resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
		# correct scaling of flow
		u, v = resized_flow.chunk(2, dim=1)
		u *= float(size_targets[1] / size_inputs[1])
		v *= float(size_targets[0] / size_inputs[0])
		return torch.cat([u, v], dim=1)

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_sf_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		teacher_dict = output_dict['teacher_dict']
		student_dict = output_dict['student_dict']

		ii = 0
		
		sf_f_student = student_dict['flow_f'][ii]
		sf_b_student = student_dict['flow_b'][ii]
		disp_l1_student = student_dict['disp_l1'][ii]
		disp_l2_student = student_dict['disp_l2'][ii]
		disp_r1_student = student_dict['output_dict_r']['disp_l1'][ii]
		disp_r2_student = student_dict['output_dict_r']['disp_l2'][ii]

		sf_f_teacher = teacher_dict['flow_f_pp'][ii]
		sf_b_teacher = teacher_dict['flow_b_pp'][ii]
		disp_l1_teacher = teacher_dict['disp_l1_pp'][ii]
		disp_l2_teacher = teacher_dict['disp_l2_pp'][ii]
		disp_r1_teacher = teacher_dict['output_dict_r']['disp_l1'][ii]
		disp_r2_teacher = teacher_dict['output_dict_r']['disp_l2'][ii]

		sf_f_teacher = self.upsample_flow_as(sf_f_teacher, sf_f_student)
		sf_b_teacher = self.upsample_flow_as(sf_b_teacher, sf_b_student)
		disp_l1_teacher = interpolate2d_as(disp_l1_teacher, disp_l1_student, mode="bilinear") * disp_l1_student.size(3)
		disp_l2_teacher = interpolate2d_as(disp_l2_teacher, disp_l2_student, mode="bilinear") * disp_l2_student.size(3)
		disp_r1_teacher = interpolate2d_as(disp_r1_teacher, disp_r1_student, mode="bilinear") * disp_r1_student.size(3)
		disp_r2_teacher = interpolate2d_as(disp_r2_teacher, disp_r2_student, mode="bilinear") * disp_r2_student.size(3)

			
		## For image reconstruction loss
		img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f_student)
		img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b_student)
		img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f_student)
		img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b_student)

		#print(disp_l1_teacher.requires_grad)

		# teacher occ maskes:
		disp_occ_l1_teacher = _adaptive_disocc_detection_disp(disp_r1_teacher).detach()
		disp_occ_l2_teacher = _adaptive_disocc_detection_disp(disp_r2_teacher).detach()
		flow_occ_f_teacher = _adaptive_disocc_detection(sf_f_teacher).detach()
		flow_occ_b_teacher = _adaptive_disocc_detection(sf_b_teacher).detach()

		## Disp Loss
		loss_disp_l1, disp_ts_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1_student, disp_r1_student, img_l1_aug, img_r1_aug, disp_occ_l1_teacher, disp_l1_teacher, 0)
		loss_disp_l2, disp_ts_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2_student, disp_r2_student, img_l2_aug, img_r2_aug, disp_occ_l2_teacher, disp_l2_teacher, 0)
		disp_ts = disp_ts_l1 + disp_ts_l2
		loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii] + disp_ts


		## Sceneflow Loss           
		loss_flow, loss_im, loss_smooth, flow_ts = self.flow_loss(sf_f_student, sf_b_student, img_l1_aug, img_l2_aug, sf_f_teacher, sf_b_teacher, flow_occ_f_teacher, flow_occ_b_teacher)



		loss_sf_sum = loss_sf_sum + loss_flow * self._weights[ii]           
		loss_sf_2d = loss_sf_2d + loss_im
		loss_sf_sm = loss_sf_sm + loss_smooth

		# finding weight
		f_loss = loss_sf_sum.detach()
		d_loss = loss_dp_sum.detach()
		max_val = torch.max(f_loss, d_loss)
		f_weight = max_val / f_loss
		d_weight = max_val / d_loss

		total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

		loss_dict = {}
		loss_dict["dp"] = loss_dp_sum
		loss_dict["flow"] = loss_sf_sum
		loss_dict["im"] = loss_sf_2d
		loss_dict["dp_ts"] = disp_ts
		loss_dict["f_ts"] = flow_ts
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(student_dict['output_dict_r'])

		return loss_dict


class Loss_MonoFlowDisp_DispC_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDisp_DispC_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		return left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_dispC_v3(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_dispC_v3(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)

		occ_map_f = _adaptive_disocc_detection(flow_f).detach()
		occ_map_b = _adaptive_disocc_detection(flow_b).detach()

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		loss_flo_f = loss_flow_f[occ_map_f].mean()
		loss_flo_b = loss_flow_b[occ_map_b].mean()
		loss_flow_f[~occ_map_f].detach_()
		loss_flow_b[~occ_map_b].detach_()
		loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = self._sf_3d_pts + loss_flo * 0.05 + loss_dispC_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_flo * 0.05, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_flos = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			#loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_flo, loss_3d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_flos = loss_flos + loss_flo
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()

		total_loss = loss_sf_sum

		loss_dict = {}
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["flo"] = loss_flos
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])
		#self.detaching_grad_of_outputs(output_dict['output_dict_r']['dispC_b'])

		return loss_dict

class Loss_MonoFlowDisp_DispC_SelfSup_v2(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDisp_DispC_SelfSup_v2, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		return left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp 

		rel_scale = local_scale / aug_size
		intrinsic_dp_s_l1 = intrinsic_scale(k_l1_aug, rel_scale[:,0], rel_scale[:,1])
		intrinsic_dp_s_l2 = intrinsic_scale(k_l2_aug, rel_scale[:,0], rel_scale[:,1])
		depthC_f = disp2depth_kitti(dispC_f / w_dp, intrinsic_dp_s_l1[:,0,0])
		depthC_b = disp2depth_kitti(dispC_b / w_dp, intrinsic_dp_s_l2[:,0,0])        

		#sf_f = flow2sf_dispC_v3(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		#sf_b = flow2sf_dispC_v3(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts1_next, k1_scale_next = pixel2pts_ms(k_l1_aug, disp_l1 + dispC_f, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)
		pts2_next, k2_scale_next = pixel2pts_ms(k_l2_aug, disp_l2 + dispC_b, local_scale / aug_size)

		#_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		#_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts1_warp = self._warping_layer(pts1_next, flow_f)
		pts2_warp = self._warping_layer(pts2_next, flow_b)

		occ_map_f = _adaptive_disocc_detection(flow_f).detach()
		occ_map_b = _adaptive_disocc_detection(flow_b).detach()

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		#loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		#loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		# loss_flo_f = loss_flow_f[occ_map_f].mean()
		# loss_flo_b = loss_flow_b[occ_map_b].mean()
		# loss_flow_f[~occ_map_f].detach_()
		# loss_flow_b[~occ_map_b].detach_()
		# loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		# expansion smoothness loss 
		loss_dispC_s = ((_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10) / disp_l1).mean() + (_smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10) / disp_l2).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		#loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = self._sf_3d_pts + loss_dispC_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_dispC_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_flos = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			#loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()

		total_loss = loss_sf_sum

		loss_dict = {}
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])
		#self.detaching_grad_of_outputs(output_dict['output_dict_r']['dispC_b'])

		return loss_dict

class Loss_MonoFlowDisp_Exp_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDisp_Exp_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		return left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_exp(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_exp(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)

		occ_map_f = _adaptive_disocc_detection(flow_f).detach()
		occ_map_b = _adaptive_disocc_detection(flow_b).detach()

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		loss_flo_f = loss_flow_f[occ_map_f].mean()
		loss_flo_b = loss_flow_b[occ_map_b].mean()
		loss_flow_f[~occ_map_f].detach_()
		loss_flow_b[~occ_map_b].detach_()
		loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = self._sf_3d_pts + loss_flo * 0.05 + loss_dispC_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_flo * 0.05, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_flos = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			#loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_flo, loss_3d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_flos = loss_flos + loss_flo
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()

		total_loss = loss_sf_sum

		loss_dict = {}
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["flo"] = loss_flos
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])
		#self.detaching_grad_of_outputs(output_dict['output_dict_r']['dispC_b'])

		return loss_dict

class Loss_MonoFlowDisp_DepthC_SelfSup(nn.Module):
	def __init__(self, args):
		super(Loss_MonoFlowDisp_DepthC_SelfSup, self).__init__()
				
		self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
		self._ssim_w = 0.85
		self._disp_smooth_w = 0.1
		self._sf_3d_pts = 0.2
		self._sf_3d_sm = 200
		self._warping_layer = WarpingLayer_Flow()

	def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

		img_r_warp = _generate_image_left(img_r_aug, disp_l)
		left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

		return left_occ


	def sceneflow_loss(self, dispC_f, dispC_b, flow_f, flow_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

		_, _, h_dp, w_dp = flow_f.size()
		disp_l1 = disp_l1 * w_dp
		disp_l2 = disp_l2 * w_dp

		## scale
		local_scale = torch.zeros_like(aug_size)
		local_scale[:, 0] = h_dp
		local_scale[:, 1] = w_dp        

		sf_f = flow2sf_depthC(flow_f, disp_l1, dispC_f, k_l1_aug, local_scale / aug_size)
		sf_b = flow2sf_depthC(flow_b, disp_l2, dispC_b, k_l2_aug, local_scale / aug_size) 

		pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
		pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

		_, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
		_, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

		pts2_warp = reconstructPts(coord1, pts2)
		pts1_warp = reconstructPts(coord2, pts1) 

		pred_flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
		pred_flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)

		occ_map_f = _adaptive_disocc_detection(flow_f).detach()
		occ_map_b = _adaptive_disocc_detection(flow_b).detach()

		#print(occ_map_f.shape)

		#print(mask_f.shape)
		#print(self.affine(flow_f).shape)
		img_l2_warp = self._warping_layer(img_l2_aug, flow_f)
		img_l1_warp = self._warping_layer(img_l1_aug, flow_b)

		## Image reconstruction loss
		img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
		loss_im1 = img_diff1[occ_map_f].mean()
		loss_im2 = img_diff2[occ_map_b].mean()
		img_diff1[~occ_map_f].detach_()
		img_diff2[~occ_map_b].detach_()
		loss_im = loss_im1 + loss_im2

		# flow_reg_loss
		loss_flow_f = _elementwise_epe(flow_f,pred_flow_f)
		loss_flow_b = _elementwise_epe(flow_b,pred_flow_b)

		loss_flo_f = loss_flow_f[occ_map_f].mean()
		loss_flo_b = loss_flow_b[occ_map_b].mean()
		loss_flow_f[~occ_map_f].detach_()
		loss_flow_b[~occ_map_b].detach_()
		loss_flo = loss_flo_f + loss_flo_b

		#print("loss_im dim", loss_im.shape)
		
		## Point reconstruction Loss
		pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
		pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

		pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
		pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
		loss_pts1 = pts_diff1[occ_map_f].mean()
		loss_pts2 = pts_diff2[occ_map_b].mean()
		pts_diff1[~occ_map_f].detach_()
		pts_diff2[~occ_map_b].detach_()
		loss_pts = loss_pts1 + loss_pts2

		#print("loss_pts dim", loss_pts.shape)

		# flow smoothness loss
		# expansion smoothness loss 
		loss_dispC_s = (_smoothness_motion_2nd(dispC_f, img_l1_aug, beta=10).mean() + _smoothness_motion_2nd(dispC_b, img_l2_aug, beta=10).mean()) / (2 ** ii)
		#print("smoothness dim", loss_flow_s.shape, loss_exp_s.shape)
		## 3D motion smoothness loss
		loss_3d_s = 0.1 * loss_dispC_s

		#loss_2d_s = _smoothness_motion_2nd(flow_f / 20.0, img_l1_aug, beta=10.0).mean() + _smoothness_motion_2nd(flow_b / 20.0, img_l2_aug, beta=10.0).mean()

		## Loss Summnation
		sceneflow_loss = self._sf_3d_pts + loss_flo * 0.05 + loss_dispC_s
		
		return sceneflow_loss, loss_im, loss_pts, loss_flo * 0.05, loss_3d_s

	def detaching_grad_of_outputs(self, output_dict):
		
		for ii in range(0, len(output_dict['flow_f'])):
			output_dict['flow_f'][ii].detach_()
			output_dict['flow_b'][ii].detach_()
			output_dict['disp_l1'][ii].detach_()
			output_dict['disp_l2'][ii].detach_()

		return None

	def forward(self, output_dict, target_dict):

		loss_dict = {}

		batch_size = target_dict['input_l1'].size(0)
		loss_sf_sum = 0
		loss_dp_sum = 0
		loss_sf_2d = 0
		loss_sf_3d = 0
		loss_flos = 0
		loss_sf_3d_sm = 0
		
		k_l1_aug = target_dict['input_k_l1_aug']
		k_l2_aug = target_dict['input_k_l2_aug']
		aug_size = target_dict['aug_size']

		disp_r1_dict = output_dict['output_dict_r']['disp_l1']
		disp_r2_dict = output_dict['output_dict_r']['disp_l2']

		for ii, (sf_f, sf_b, dispC_f, dispC_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['dispC_f'], output_dict['dispC_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

			assert(sf_f.size()[2:4] == sf_b.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
			assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
			
			## For image reconstruction loss
			img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
			img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
			img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
			img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

			## Disp Loss
			disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
			disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
			#loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


			## Sceneflow Loss           
			loss_sceneflow, loss_im, loss_pts, loss_flo, loss_3d_s = self.sceneflow_loss(dispC_f, dispC_b, sf_f, sf_b, 
																			disp_l1, disp_l2,
																			disp_occ_l1, disp_occ_l2,
																			k_l1_aug, k_l2_aug,
																			img_l1_aug, img_l2_aug, 
																			aug_size, ii)

			loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
			loss_sf_2d = loss_sf_2d + loss_im            
			loss_sf_3d = loss_sf_3d + loss_pts
			loss_flos = loss_flos + loss_flo
			loss_sf_3d_sm = loss_sf_3d_sm + loss_3d_s


		# finding weight
		f_loss = loss_sf_sum.detach()

		total_loss = loss_sf_sum

		loss_dict = {}
		loss_dict["sf"] = loss_sf_sum
		loss_dict["s_2"] = loss_sf_2d
		loss_dict["s_3"] = loss_sf_3d
		loss_dict["flo"] = loss_flos
		loss_dict["s_3s"] = loss_sf_3d_sm
		loss_dict["total_loss"] = total_loss

		self.detaching_grad_of_outputs(output_dict['output_dict_r'])
		#self.detaching_grad_of_outputs(output_dict['output_dict_r']['dispC_b'])

		return loss_dict

