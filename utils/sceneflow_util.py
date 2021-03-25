from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torch.nn.functional as tf


def post_processing(l_disp, r_disp):
	
	b, _, h, w = l_disp.shape
	m_disp = 0.5 * (l_disp + r_disp)
	grid_l = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w).float().requires_grad_(False).cuda()
	l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), 0, 1)
	r_mask = torch.flip(l_mask, [3])
	return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def flow_horizontal_flip(flow_input):

	flow_flip = torch.flip(flow_input, [3])
	flow_flip[:, 0:1, :, :] *= -1

	return flow_flip.contiguous()


def disp2depth_kitti(pred_disp, k_value):

	pred_depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-8)
	pred_depth = torch.clamp(pred_depth, 1e-3, 80)

	return pred_depth


def get_pixelgrid(b, h, w):
	grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
	grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

	ones = torch.ones_like(grid_h)
	pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()

	return pixelgrid


def pixel2pts(intrinsics, depth):
	b, _, h, w = depth.size()

	pixelgrid = get_pixelgrid(b, h, w)

	depth_mat = depth.view(b, 1, -1)
	pixel_mat = pixelgrid.view(b, 3, -1)
	#print("pixelgrid shape:",pixel_mat.shape)

	pts_mat = torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat
	pts = pts_mat.view(b, -1, h, w)

	return pts, pixelgrid

def pixel2pts_disp(intrinsics, depth, pixelgrid):
	b, _, h, w = depth.size()

	#pixelgrid = get_pixelgrid(b, h, w)

	depth_mat = depth.view(b, 1, -1)
	pixel_mat = pixelgrid.view(b, 3, -1)
	#print("pixelgrid shape:",pixel_mat.shape)

	pts_mat = torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat
	pts = pts_mat.view(b, -1, h, w)

	return pts


def pts2pixel(pts, intrinsics):
	b, _, h, w = pts.size()
	proj_pts = torch.matmul(intrinsics, pts.view(b, 3, -1))
	pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]

	return pixels_mat.view(b, 2, h, w)

def flow2sf(flow, disp, exp, intrinsic, rel_scale):
	b, _, h, w = disp.size()
	disp_next = disp*(1 - torch.exp(exp))
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	depth = disp2depth_kitti(disp, intrinsic_dp_s[:,0,0])
	depth_next = disp2depth_kitti(disp_next, intrinsic_dp_s[:,0,0])

	grid = get_pixelgrid(b,h,w)
	#print(grid.shape)
	grid_next = torch.cat((grid[:,:2,:,:] + flow,torch.ones_like(disp)),dim=1)

	pts = pixel2pts_disp(intrinsic_dp_s, depth, grid)
	pts_next = pixel2pts_disp(intrinsic_dp_s, depth_next, grid_next)

	return pts_next - pts

def flow2sf_exp(flow, disp, exp, intrinsic, rel_scale):
	b, _, h, w = disp.size()
	disp_next = disp/torch.exp(exp)
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	depth = disp2depth_kitti(disp, intrinsic_dp_s[:,0,0])
	depth_next = disp2depth_kitti(disp_next, intrinsic_dp_s[:,0,0])

	grid = get_pixelgrid(b,h,w)
	#print(grid.shape)
	grid_next = torch.cat((grid[:,:2,:,:] + flow,torch.ones_like(disp)),dim=1)

	pts = pixel2pts_disp(intrinsic_dp_s, depth, grid)
	pts_next = pixel2pts_disp(intrinsic_dp_s, depth_next, grid_next)

	return pts_next - pts

def flow2sf_dispC(flow, disp, dispC, intrinsic, rel_scale):
	b, _, h, w = disp.size()
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	depth = disp2depth_kitti(disp, intrinsic_dp_s[:,0,0])
	flow_x = flow[:,0:1,:,:] - dispC
	depthC = disp2depth_kitti(dispC*w, intrinsic_dp_s[:,0,0])
	#print(intrinsic_dp_s[:,0,0].shape)
	#print(flow_x.shape)
	sf_x = flow_x / intrinsic_dp_s[:,0,0].view(b,1,1,1) * depth
	sf_y = flow[:,1:2,:,:] / intrinsic_dp_s[:,1,1].view(b,1,1,1) * depth 
	sf = torch.cat((sf_x, sf_y, depthC),dim=1)
	return sf

def flow2sf_dispC_v2(flow, disp, dispC, intrinsic, rel_scale):
	b, _, h, w = disp.size()
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	depth = disp2depth_kitti(disp, intrinsic_dp_s[:,0,0])
	flow_x = flow[:,0:1,:,:] - dispC
	depthC = disp2depth_kitti(dispC, intrinsic_dp_s[:,0,0])
	#print(intrinsic_dp_s[:,0,0].shape)
	#print(flow_x.shape)
	sf_x = flow_x / intrinsic_dp_s[:,0,0].view(b,1,1,1) * depth
	sf_y = flow[:,1:2,:,:] / intrinsic_dp_s[:,1,1].view(b,1,1,1) * depth 
	sf = torch.cat((sf_x, sf_y, depthC),dim=1)
	return sf

def flow2sf_dispC_v3(flow, disp, dispC, intrinsic, rel_scale):
	b, _, h, w = disp.size()
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	depth = disp2depth_kitti(disp, intrinsic_dp_s[:,0,0])
	flow_x = flow[:,0:1,:,:] + dispC * w
	depthC = disp2depth_kitti(dispC * w, intrinsic_dp_s[:,0,0])
	#print(intrinsic_dp_s[:,0,0].shape)
	#print(flow_x.shape)
	sf_x = flow_x / intrinsic_dp_s[:,0,0].view(b,1,1,1) * depth
	sf_y = flow[:,1:2,:,:] / intrinsic_dp_s[:,1,1].view(b,1,1,1) * depth 
	sf = torch.cat((sf_x, sf_y, depthC),dim=1)
	return sf


def intrinsic_scale(intrinsic, scale_y, scale_x):
	b, h, w = intrinsic.size()
	fx = intrinsic[:, 0, 0] * scale_x
	fy = intrinsic[:, 1, 1] * scale_y
	cx = intrinsic[:, 0, 2] * scale_x
	cy = intrinsic[:, 1, 2] * scale_y

	zeros = torch.zeros_like(fx)
	r1 = torch.stack([fx, zeros, cx], dim=1)
	r2 = torch.stack([zeros, fy, cy], dim=1)
	r3 = torch.tensor([0., 0., 1.], requires_grad=False).cuda().unsqueeze(0).expand(b, -1)
	intrinsic_s = torch.stack([r1, r2, r3], dim=1)

	return intrinsic_s


def pixel2pts_ms(intrinsic, output_disp, rel_scale):
	# pixel2pts
	intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
	output_depth = disp2depth_kitti(output_disp, intrinsic_dp_s[:, 0, 0])
	pts, _ = pixel2pts(intrinsic_dp_s, output_depth)
	#print("pts shape:",)

	return pts, intrinsic_dp_s


def pts2pixel_ms(intrinsic, pts, output_sf, disp_size):
	b, _, h, w = output_sf.size()
	# +sceneflow and reprojection
	sf_s = tf.interpolate(output_sf, disp_size, mode="bilinear", align_corners=True)
	pts_tform = pts + sf_s
	coord = pts2pixel(pts_tform, intrinsic)
	flow = coord - get_pixelgrid(b, h, w)[:, 0:2, :, :]

	norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
	norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
	norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)

	return flow[:, 0:2, :, :], pts_tform, norm_coord

def get_grid(x):
	grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
	grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
	grid = torch.cat([grid_H, grid_V], 1)
	grids_cuda = grid.float().requires_grad_(False).cuda()
	return grids_cuda

def warp_Img_flow(x, flow):
	flo_list = []
	flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
	flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
	flo_list.append(flo_w)
	flo_list.append(flo_h)
	flow_for_grid = torch.stack(flo_list).transpose(0, 1)
	grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
	x_warp = tf.grid_sample(x, grid)

	mask = torch.ones(x.size(), requires_grad=False).cuda()
	mask = tf.grid_sample(mask, grid)
	mask = (mask >= 1.0).float()

	return x_warp * mask


def reconstructImg(coord, img):
	grid = coord.transpose(1, 2).transpose(2, 3)
	img_warp = tf.grid_sample(img, grid)

	mask = torch.ones_like(img, requires_grad=False)
	mask = tf.grid_sample(mask, grid)
	mask = (mask >= 1.0).float()
	return img_warp * mask


def reconstructPts(coord, pts):
	#print("pts shape:", pts.shape)
	#print("coord shape:", coord.shape)
	grid = coord.transpose(1, 2).transpose(2, 3)
	pts_warp = tf.grid_sample(pts, grid)

	mask = torch.ones_like(pts, requires_grad=False)
	mask = tf.grid_sample(mask, grid)
	mask = (mask >= 1.0).float()
	return pts_warp * mask


def projectSceneFlow2Flow(intrinsic, sceneflow, disp):

	_, _, h, w = disp.size()

	output_depth = disp2depth_kitti(disp, intrinsic[:, 0, 0])
	pts, pixelgrid = pixel2pts(intrinsic, output_depth)

	sf_s = tf.interpolate(sceneflow, [h, w], mode="bilinear", align_corners=True)
	pts_tform = pts + sf_s
	coord = pts2pixel(pts_tform, intrinsic)
	flow = coord - pixelgrid[:, 0:2, :, :]

	return flow
