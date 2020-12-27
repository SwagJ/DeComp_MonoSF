from __future__ import absolute_import, division, print_function

import os.path
import torch
import numpy as np
import skimage.io as io
import png
import cv2
import random
import math

width_to_date = dict()
width_to_date[1242] = '2011_09_26'
width_to_date[1224] = '2011_09_28'
width_to_date[1238] = '2011_09_29'
width_to_date[1226] = '2011_09_30'
width_to_date[1241] = '2011_10_03'


def get_date_from_width(width):
	return width_to_date[width]


def list_flatten(input_list):
	return [img for sub_list in input_list for img in sub_list]


def intrinsic_scale(mat, sy, sx):
	out = mat.clone()
	out[0, 0] *= sx
	out[0, 2] *= sx
	out[1, 1] *= sy
	out[1, 2] *= sy
	return out


def kitti_adjust_intrinsic(k_l1, k_r1, crop_info):
	str_x = crop_info[0]
	str_y = crop_info[1]
	k_l1[0, 2] -= str_x
	k_l1[1, 2] -= str_y
	k_r1[0, 2] -= str_x
	k_r1[1, 2] -= str_y
	return k_l1, k_r1

def kitti_crop_image_list(img_list, crop_info):    
	str_x = crop_info[0]
	str_y = crop_info[1]
	end_x = crop_info[2]
	end_y = crop_info[3]

	transformed = [img[str_y:end_y, str_x:end_x, :] for img in img_list]

	return transformed


def numpy2torch(array):
	assert(isinstance(array, np.ndarray))
	if array.ndim == 3:
		array = np.transpose(array, (2, 0, 1))
	else:
		array = np.expand_dims(array, axis=0)
	return torch.from_numpy(array.copy()).float()


def read_image_as_byte(filename):
	return io.imread(filename)


def read_png_flow(flow_file):
	flow_object = png.Reader(filename=flow_file)
	flow_direct = flow_object.asDirect()
	flow_data = list(flow_direct[2])
	(w, h) = flow_direct[3]['size']
	flow = np.zeros((h, w, 3), dtype=np.float64)
	for i in range(len(flow_data)):
		flow[i, :, 0] = flow_data[i][0::3]
		flow[i, :, 1] = flow_data[i][1::3]
		flow[i, :, 2] = flow_data[i][2::3]

	invalid_idx = (flow[:, :, 2] == 0)
	flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
	flow[invalid_idx, 0] = 0
	flow[invalid_idx, 1] = 0
	return flow[:, :, 0:2], (1 - invalid_idx * 1)[:, :, None]


def read_png_disp(disp_file):
	disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
	disp_np = np.expand_dims(disp_np, axis=2)
	mask_disp = (disp_np > 0).astype(np.float64)
	return disp_np, mask_disp

		
def read_raw_calib_file(filepath):
	# From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
	"""Read in a calibration file and parse into a dictionary."""
	data = {}

	with open(filepath, 'r') as f:
		for line in f.readlines():
			key, value = line.split(':', 1)
			# The only non-float values in these files are dates, which
			# we don't care about anyway
			try:
				data[key] = np.array([float(x) for x in value.split()])
			except ValueError:
				pass
	return data

def get_r_matrix(rx=0,ry=0,rz=0):
	theta_x = 2*math.pi * rx / 360
	theta_y = 2*math.pi * ry / 360
	theta_z = 2*math.pi * rz / 360

	Ry = np.array([[1,0,0],
				[0,math.cos(theta_x),-math.sin(theta_x)],
				[0,math.sin(theta_x),math.cos(theta_x)]])
	Rx = np.array([[math.cos(theta_y),0,math.sin(theta_y)],
				[0,1,0],
				[-math.sin(theta_y),0,math.cos(theta_y)]]
				)
	Rz = np.array([[math.cos(theta_z),-math.sin(theta_z),0],
				[math.sin(theta_z),math.cos(theta_z),0],
				[0,0,1]])

	total_R = np.matmul(np.matmul(Rz,Ry),Rx)

	return total_R


def read_calib_into_dict(path_dir):

	calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
	intrinsic_dict_l = {}
	intrinsic_dict_r = {}

	for ii, date in enumerate(calibration_file_list):
		file_name = "cam_intrinsics/calib_cam_to_cam_" + date + '.txt'
		file_name_full = os.path.join(path_dir, file_name)
		file_data = read_raw_calib_file(file_name_full)
		P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
		P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
		intrinsic_dict_l[date] = P_rect_02[:3, :3]
		intrinsic_dict_r[date] = P_rect_03[:3, :3]

	return intrinsic_dict_l, intrinsic_dict_r

def pixel_coord_np(width, height):
	x = np.linspace(0, width - 1, width).astype(np.int)
	y = np.linspace(0, height - 1, height).astype(np.int)
	[x, y] = np.meshgrid(x, y)
	return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def get_shear_x(sy=0,sz=0):
	Shx = np.array([[1,0,0,0],
				[sy,1,0,0],
				[sz,0,1,0],
				[0,0,0,1]])

	return Shx

def get_shear_y(sx=0,sz=0):
	Shy = np.array([[1,sx,0,0],
				[0,1,0,0],
				[0,sz,1,0],
				[0,0,0,1]])
	return Shy

def get_shear_z(sx=0,sy=0):
	Shz = np.array([[1,0,sx,0],
				[0,1,sy,0],
				[0,0,1,0],
				[0,0,0,1]])
	return Shz

def get_valid_map(new_coords,h,w):
	return (new_coords[:,0] < w) * (new_coords[:,0] >= 0) * (new_coords[:,1] < h) * (new_coords[:,1] >= 0) 

def perform_shearing(cam_coords_l,cam_coords_r):
	sh = get_shear_y(random.uniform(-0.05,0.05),random.uniform(-0.05,0.05))
	#expand input dim
	ones_l = np.ones((cam_coords_l.shape[0],1))
	cam_coords_l = np.concatenate((cam_coords_l,ones_l),axis=1)
	ones_r = np.ones((cam_coords_r.shape[0],1))
	cam_coords_r = np.concatenate((cam_coords_r,ones_r),axis=1)

	shed_cam_coords_l = np.matmul(sh,cam_coords_l.T).T
	shed_cam_coords_r = np.matmul(sh,cam_coords_r.T).T

	return shed_cam_coords_l[:,:3],shed_cam_coords_r[:,:3]

def perform_rotate(cam_coords_l,cam_coords_r):
	R = get_r_matrix(0,random.uniform(-8,8),0)
	return np.matmul(cam_coords_l,R),np.matmul(cam_coords_r,R)

def perform_trans(cam_coords_l,cam_coords_r,depth_min):
	tz = random.uniform(-depth_min,0)
	T = [0,0,tz]
	valid_area_l = cam_coords_l[:,2] >= -T[-1]
	cam_coords_l[valid_area_l,-1] = cam_coords_l[valid_area_l,-1] + T[-1]
	valid_area_r = cam_coords_r[:,2] >= -T[-1]
	cam_coords_r[valid_area_r,-1] = cam_coords_r[valid_area_r,-1] + T[-1]
	return cam_coords_l,cam_coords_r

def perform_scaling(cam_coords_l,cam_coords_r):
	scaling_x = random.uniform(0,0.05)
	scaling_y = random.uniform(0,0.05)
	scaling_z = random.uniform(0,0.05)
	cam_coords_l[:,0] = cam_coords_l[:,0] * scaling_x
	cam_coords_l[:,1] = cam_coords_l[:,1] * scaling_y
	cam_coords_l[:,2] = cam_coords_l[:,2] * scaling_z
	cam_coords_r[:,0] = cam_coords_r[:,0] * scaling_x
	cam_coords_r[:,1] = cam_coords_r[:,1] * scaling_y
	cam_coords_r[:,2] = cam_coords_r[:,2] * scaling_z
	return cam_coords_l,cam_coords_r

def threed_warp(image_l,depth_l,image_r,depth_r,intrinsic_l,intrinsic_r):
	height_l = depth_l.shape[0]
	width_l = depth_l.shape[1]
	height_r = depth_r.shape[0]
	width_r = depth_r.shape[1]
	depth_min = np.min(np.vstack((depth_l,depth_r)))

	inv_intrinsic_l = np.linalg.inv(intrinsic_l)
	inv_intrinsic_r = np.linalg.inv(intrinsic_r)

	image_coord_l = pixel_coord_np(width_l,height_l)
	cam_coords_l = (inv_intrinsic_l[:3, :3] @ image_coord_l * depth_l.flatten()).T
	image_coord_r = pixel_coord_np(width_r,height_r)
	cam_coords_r = (inv_intrinsic_l[:3, :3] @ image_coord_r * depth_r.flatten()).T

	# rotation
	if torch.rand(1) > 0.5:
		aug_cam_coords_l,aug_cam_coords_r = perform_rotate(cam_coords_l,cam_coords_r)
	else: 
		aug_cam_coords_l,aug_cam_coords_r = perform_scaling(cam_coords_l,cam_coords_r)
	# shearing
	aug_cam_coords_l,aug_cam_coords_r = perform_shearing(aug_cam_coords_l,aug_cam_coords_r)
	#translation
	aug_cam_coords_l,aug_cam_coords_r = perform_trans(aug_cam_coords_l,aug_cam_coords_r,depth_min)
	# generate sceneflow
	sf_l = (aug_cam_coords_l - cam_coords_l).reshape(height_l,width_l,3).astype(np.float32)
	sf_r = (aug_cam_coords_r - cam_coords_r).reshape(height_r,width_r,3).astype(np.float32)
	#get warpped image
	new_image_coords_l = (intrinsic_l @ aug_cam_coords_l.T / (aug_cam_coords_l[:,2]+ 1e-6)).T
	new_image_coords_r = (intrinsic_r @ aug_cam_coords_r.T / (aug_cam_coords_r[:,2]+ 1e-6)).T
	#get valid map
	valid_l = get_valid_map(new_image_coords_l[:,:2],height_l,width_l).reshape(height_l,width_l).astype(np.float32)
	valid_r = get_valid_map(new_image_coords_r[:,:2],height_r,width_r).reshape(height_r,width_r).astype(np.float32)
	# get new image coordinate
	new_img_xl = new_image_coords_l[:,0].reshape(height_l,width_l).astype(np.float32)
	new_img_yl = new_image_coords_l[:,1].reshape(height_l,width_l).astype(np.float32)
	new_img_xr = new_image_coords_r[:,0].reshape(height_r,width_r).astype(np.float32)
	new_img_yr = new_image_coords_r[:,1].reshape(height_r,width_r).astype(np.float32)

	warpped_im_l = cv2.remap(image_l,new_img_xl,new_img_yl,cv2.INTER_LINEAR)
	valid_pixels_l = (warpped_im_l != 0).all(-1).astype(np.float32)
	warpped_im_r = cv2.remap(image_r,new_img_xr,new_img_yr,cv2.INTER_LINEAR)
	valid_pixels_r = (warpped_im_r != 0).all(-1).astype(np.float32)

	sf_bl = -cv2.remap(sf_l,new_img_xl,new_img_yl,cv2.INTER_LINEAR)
	sf_br = -cv2.remap(sf_r,new_img_xr,new_img_yr,cv2.INTER_LINEAR)
	#cv2.imwrite("l.png", warpped_im_l)
	#cv2.imwrite("r.png", warpped_im_r)
	#cv2.imwrite("l_og.png", image_l)
	#cv2.imwrite("r_og.png", image_r)

	out_dict = {
		"sf_l":sf_l,
		"sf_r":sf_r,
		"sf_bl":sf_bl,
		"sf_br":sf_br,
		"warpped_im_l":warpped_im_l,
		"warpped_im_r":warpped_im_r,
		"valid_l":np.expand_dims(valid_l, axis=2),
		"valid_r":np.expand_dims(valid_r, axis=2),
		"valid_pixels_l":np.expand_dims(valid_pixels_l,axis=2),
		"valid_pixels_r":np.expand_dims(valid_pixels_r,axis=2)
	}
	return out_dict