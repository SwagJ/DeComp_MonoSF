from __future__ import absolute_import, division, print_function

import os.path
import torch
import numpy as np
import skimage.io as io
import png
import cv2
import random
import math
from . import exp_aug as flow_transforms
from PIL import Image
import torch.nn.functional as tf

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

def read_png_file(flow_file):
	"""
	Read from KITTI .png file
	:param flow_file: name of the flow file
	:return: optical flow data in matrix
	"""
	flow = cv2.imread(flow_file,-1)[:,:,::-1].astype(np.float64)
 #   flow_object = png.Reader(filename=flow_file)
 #   flow_direct = flow_object.asDirect()
 #   flow_data = list(flow_direct[2])
 #   (w, h) = flow_direct[3]['size']
 #   #print("Reading %d x %d flow file in .png format" % (h, w))
 #   flow = np.zeros((h, w, 3), dtype=np.float64)
 #   for i in range(len(flow_data)):
 #       flow[i, :, 0] = flow_data[i][0::3]
 #       flow[i, :, 1] = flow_data[i][1::3]
 #       flow[i, :, 2] = flow_data[i][2::3]

	invalid_idx = (flow[:, :, 2] == 0)
	flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
	flow[invalid_idx, 0] = 0
	flow[invalid_idx, 1] = 0
	return flow

def read_png_disp(disp_file):
	disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
	disp_np = np.expand_dims(disp_np, axis=2)
	mask_disp = (disp_np > 0).astype(np.float64)
	return disp_np, mask_disp

def read_full_disp(path):
	data = Image.open(path)
	data = np.ascontiguousarray(data,dtype=np.float32)/256
	return data

		
def read_raw_calib_file(filepath):
	# From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
	"""Read in a calibration file and parse into a dictionary."""
	data = {}
	#print(filepath)
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

def read_full_calib_into_dict(path_dir):

	calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
	intrinsic_dict_l = {}
	intrinsic_dict_r = {}

	for ii, date in enumerate(calibration_file_list):
		file_name = "cam_intrinsics/calib_cam_to_cam_" + date + '.txt'
		file_name_full = os.path.join(path_dir, file_name)
		file_data = read_raw_calib_file(file_name_full)
		P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
		P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
		intrinsic_dict_l[date] = P_rect_02
		intrinsic_dict_r[date] = P_rect_03

	return intrinsic_dict_l, intrinsic_dict_r

def process_calib_into_dict(P_rect_20, P_rect_30):
	data = {}
	data['K_cam2'] = P_rect_20[0:3, 0:3]
	data['K_cam3'] = P_rect_30[0:3, 0:3]
	data['b20'] = P_rect_20[0, 3] / P_rect_20[0, 0]
	data['b30'] = P_rect_30[0, 3] / P_rect_30[0, 0]
	return data


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
	if torch.rand(1) > 0.1:
		aug_cam_coords_l,aug_cam_coords_r = perform_rotate(cam_coords_l,cam_coords_r)
	else: 
		aug_cam_coords_l,aug_cam_coords_r = perform_scaling(cam_coords_l,cam_coords_r)
	# shearing
	# aug_cam_coords_l,aug_cam_coords_r = perform_shearing(aug_cam_coords_l,aug_cam_coords_r)
	#translation
	aug_cam_coords_l,aug_cam_coords_r = perform_trans(aug_cam_coords_l,aug_cam_coords_r,depth_min)
	# generate sceneflow
	sf_l = (aug_cam_coords_l - cam_coords_l).reshape(height_l,width_l,3).astype(np.float32)
	sf_r = (aug_cam_coords_r - cam_coords_r).reshape(height_r,width_r,3).astype(np.float32)
	#get warpped image
	new_image_coords_l = (intrinsic_l @ aug_cam_coords_l.T / (aug_cam_coords_l[:,2]+ 1e-6)).T
	new_image_coords_r = (intrinsic_r @ aug_cam_coords_r.T / (aug_cam_coords_r[:,2]+ 1e-6)).T
	new_image_coords_tensor_l = torch.from_numpy(new_image_coords_l[:,:2]).view(height_l,width_l,2)
	new_image_coords_l_x = new_image_coords_tensor_l[:,:,0:1] / (width_l - 1) * 2 - 1
	new_image_coords_l_y = new_image_coords_tensor_l[:,:,1:2] / (height_l - 1) * 2 - 1
	normed_image_coords_l = torch.cat((new_image_coords_l_x, new_image_coords_l_y),dim=2)
	new_image_coords_tensor_r = torch.from_numpy(new_image_coords_r[:,:2]).view(height_r,width_r,2)
	new_image_coords_r_x = new_image_coords_tensor_r[:,:,0:1] / (width_r - 1) * 2 - 1
	new_image_coords_r_y = new_image_coords_tensor_r[:,:,1:2] / (height_r - 1) * 2 - 1
	normed_image_coords_r = torch.cat((new_image_coords_r_x, new_image_coords_r_y),dim=2)
	# transfer to tensor:
	image_l = torch.from_numpy(image_l.astype(float).transpose(2,0,1))
	image_r = torch.from_numpy(image_r.astype(float).transpose(2,0,1))
	image_l_mask = torch.ones_like(image_l)
	image_r_mask = torch.ones_like(image_r)
	# warpping
	warpped_im_l = tf.grid_sample(image_l.unsqueeze(0),normed_image_coords_l.unsqueeze(0))
	warpped_im_r = tf.grid_sample(image_r.unsqueeze(0),normed_image_coords_r.unsqueeze(0))
	warpped_mask_l = (tf.grid_sample(image_l_mask.unsqueeze(0), normed_image_coords_l.unsqueeze(0)) >= 1)
	warpped_mask_r = (tf.grid_sample(image_r_mask.unsqueeze(0), normed_image_coords_r.unsqueeze(0)) >= 1)

	warpped_im_l = (warpped_im_l.float() * warpped_mask_l.float()).squeeze(0).numpy().transpose(1,2,0)
	warpped_im_r = (warpped_im_r.float() * warpped_mask_r.float()).squeeze(0).numpy().transpose(1,2,0)
	warpped_mask_l = warpped_mask_l.squeeze(0).numpy().transpose(1,2,0).all(-1)[:,:,np.newaxis]
	warpped_mask_r = warpped_mask_r.squeeze(0).numpy().transpose(1,2,0).all(-1)[:,:,np.newaxis]


	# get new image coordinate


	#sf_bl = -cv2.remap(sf_l,new_img_xl,new_img_yl,cv2.INTER_LINEAR)
	#sf_br = -cv2.remap(sf_r,new_img_xr,new_img_yr,cv2.INTER_LINEAR)
	#cv2.imwrite("l.png", warpped_im_l)
	#cv2.imwrite("r.png", warpped_im_r)
	#cv2.imwrite("l_og.png", image_l)
	#cv2.imwrite("r_og.png", image_r)

	out_dict = {
		"sf_l":sf_l,
		"sf_r":sf_r,
		"warpped_im_l":warpped_im_l,
		"warpped_im_r":warpped_im_r,
		"valid_l": warpped_mask_l,
		"valid_r": warpped_mask_r
	}
	return out_dict

def readPFM(file):
	import re
	file = open(file, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip()
	if header == b'PF':
		color = True
	elif header == b'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0: # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>' # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = np.reshape(data, shape)
	data = np.flipud(data)
	return data, scale

def generate_gt_expansion(iml0,iml1,flowl0,d1, d2, bl, fl, cx, cy, count_path=None, order=1, prob=1): 
	np.ascontiguousarray(flowl0,dtype=np.float32)
	flowl0[np.isnan(flowl0)] = 1e6 
	#print(flowl0.shape)	
	th,tw,_ = iml0.shape

	flowl0[:,:,2] = np.logical_and(np.logical_and(flowl0[:,:,2]==1, d1!=0), d2!=0).astype(float)
	shape = d1.shape
	mesh = np.meshgrid(range(shape[1]),range(shape[0]))
	xcoord = mesh[0].astype(float)
	ycoord = mesh[1].astype(float)

	P0 = triangulation(d1, xcoord, ycoord, bl=1, fl = fl, cx = cx, cy = cy)
	P1 = triangulation(d2, xcoord + flowl0[:,:,0], ycoord + flowl0[:,:,1], bl=1, fl = fl, cx = cx, cy = cy)
	dis0 = P0[2]
	dis1 = P1[2]

	change_size =  dis0.reshape(shape).astype(np.float32)
	flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))

	gt_normal = np.concatenate((d1[:,:,np.newaxis],d2[:,:,np.newaxis],d2[:,:,np.newaxis]),-1)
	change_size = np.concatenate((change_size[:,:,np.newaxis],gt_normal,flow3d),2)

	iml1 = (iml1)/255.
	iml0 = (iml0)/255.
	iml0 = iml0[:,:,::-1].copy()
	iml1 = iml1[:,:,::-1].copy()

	try:
		with open(count_path, 'r') as f:
			iter_counts = int(f.readline())
	except:
		iter_counts = 0


	schedule = [0.5, 1., 50000.]  # initial coeff, final_coeff, half life
	schedule_coeff = schedule[0] + (schedule[1] - schedule[0]) * (2/(1+np.exp(-1.0986*iter_counts/schedule[2])) - 1)

	if np.random.binomial(1,prob):
		co_transform1 = flow_transforms.Compose([
						flow_transforms.SpatialAug([th,tw],
										scale=[0.2,0.,0.1],
										rot=[0.4,0.],
										trans=[0.4,0.],
										squeeze=[0.3,0.], schedule_coeff=schedule_coeff, order=order),
		])
	else:
		co_transform1 = flow_transforms.Compose([
		flow_transforms.RandomCrop([th,tw]),
		])

	co_transform2 = flow_transforms.Compose([
		flow_transforms.pseudoPCAAug( schedule_coeff=schedule_coeff),
		#flow_transforms.PCAAug(schedule_coeff=schedule_coeff),
		flow_transforms.ChromaticAug( schedule_coeff=schedule_coeff, noise=0.06),
		])

	flowl0 = np.concatenate([flowl0,change_size],-1)
	augmented,flowl0,intr = co_transform1([iml0, iml1], flowl0, [fl,cx,cy,bl])
	imol0 = augmented[0]
	imol1 = augmented[1]
	augmented,flowl0,intr = co_transform2(augmented, flowl0, intr)

	iml0 = augmented[0]
	iml1 = augmented[1]
	flowl0 = flowl0.astype(np.float32)
	change_size = flowl0[:,:,3:]
	flowl0 = flowl0[:,:,:3]

	# randomly cover a region
	sx=0;sy=0;cx=0;cy=0
	#if np.random.binomial(1,0.5):
	#	sx = int(np.random.uniform(25,100))
	#	sy = int(np.random.uniform(25,100))
		#sx = int(np.random.uniform(50,150))
		#sy = int(np.random.uniform(50,150))
	#	cx = int(np.random.uniform(sx,iml1.shape[0]-sx))
	#	cy = int(np.random.uniform(sy,iml1.shape[1]-sy))
	#	iml1[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(iml1,0),0)[np.newaxis,np.newaxis]

	#iml0_bgr = iml0[...,::-1].copy()
	#iml1_bgr = iml1[...,::-1].copy()

	return iml0, iml1, flowl0, change_size, intr, imol0, imol1, np.asarray([cx-sx,cx+sx,cy-sy,cy+sy])

def triangulation(disp, xcoord, ycoord, bl=1, fl = 450, cx = 479.5, cy = 269.5):
	mask = disp != 0
	depth = bl*fl / (disp+1e-16) # 450px->15mm focal length
	depth = depth * mask
	#print(depth.shape, type(depth))
	X = (xcoord - cx) * depth / fl
	Y = (ycoord - cy) * depth / fl
	Z = depth
	P = np.concatenate((X[np.newaxis],Y[np.newaxis],Z[np.newaxis]),0).reshape(3,-1)
	P = np.concatenate((P,np.ones((1,P.shape[-1]))),0)
	return P