from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, read_png_flow, read_png_disp, numpy2torch
from .common import kitti_crop_image_list, kitti_adjust_intrinsic, intrinsic_scale, get_date_from_width
from .common import list_flatten, threed_warp, readPFM, generate_gt_expansion
from PIL import Image, ImageOps

import torch.nn.functional as tf
import glob
from torch.utils.data.dataset import ConcatDataset
from . import exp_aug as flow_transforms
import cv2
import math


class Synth_Driving(data.Dataset):
	def __init__(self,
				 args,
				 images_root=None,
				 datashape=[256, 704],
				 num_examples=-1,
				 fast_set=False,
				 focal_15mm=True,
				 forward_set=True,
				 left_cam=True):

		self._args = args
		self._seq_len = 1	
		self._datashape = datashape	

		## loading image -----------------------------------
		if not os.path.isdir(images_root):
			raise ValueError("Image directory '%s' not found!")

		if focal_15mm == True:
			subfolder = '15mm_focallength/'
			intrinsic = np.array([[450,0,479.5],[0,450,269.5],[0,0,1.0]])
		else:
			subfolder = '35mm_focallength/'
			intrinsic = np.array([[1050,0,479.5],[0,1050,269.5],[0,0,1.0]])

		if fast_set == True:
			total_img_num = 300
			freq_foldname = 'fast/'
		else:
			total_img_num = 800
			freq_foldname = 'slow/'

		if forward_set == True:
			dir_setname = 'scene_forwards/'
		else:
			dir_setname = 'scene_backwards/'

		if left_cam == True:
			stereo_set = 'left/'
			flow_postfix = '_L'
		else:
			stereo_set = 'right/'
			flow_postfix = '_R' 

		# iml0, iml1, imr0, imr1
		self._image_list = []
		# displ0, dispr0, dispC_l0_future, dispC_r0_future, flowl0_future, flowr0_future
		self._future_list = []
		# displ1, dispr1, dispC_l1_past, dispC_r1_past, flowl1_past, flowr1_past
		self._past_list = []

		image_folder = 'frames_cleanpass/' + subfolder
		disp_folder = 'disparity/' + subfolder
		disp_change_folder = 'disparity_change/' + subfolder
		flow_folder = 'optical_flow/' + subfolder
		#print(images_root)
		
		# future set
		for idx in range(1,total_img_num):
			#print(idx)
			idx_src = '%.4d' % (int(idx))
			idx_tgt = '%.4d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, image_folder, dir_setname, freq_foldname, stereo_set , idx_src) + '.png'
			name_l2 = os.path.join(images_root, image_folder, dir_setname, freq_foldname, stereo_set, idx_tgt) + '.png'
			disp_l1 = os.path.join(images_root, disp_folder, dir_setname, freq_foldname, stereo_set, idx_src) + '.pfm'
			disp_l2 = os.path.join(images_root, disp_folder, dir_setname, freq_foldname, stereo_set, idx_tgt) + '.pfm'
			#disp_r1 = os.path.join(images_root, disp_folder, dir_setname, freq_foldname, stereo_set, idx_src) + '.pfm'
			#disp_r2 = os.path.join(images_root, disp_folder, dir_setname, freq_foldname, stereo_set, idx_tgt) + '.pfm'
			flow_l1_future = os.path.join(images_root, flow_folder, dir_setname, freq_foldname, 'into_future/', stereo_set) + 'OpticalFlowIntoFuture_' + idx_src + flow_postfix + '.pfm'
			#flow_r1_future = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_future/', stereo_set) + 'OpticalFlowIntoFuture_' + idx_src + '_R.pfm'
			flow_l2_past = os.path.join(images_root, flow_folder, dir_setname, freq_foldname, 'into_past/', stereo_set) + 'OpticalFlowIntoPast_' + idx_tgt + flow_postfix + '.pfm'
			#flow_r2_past = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_past/', stereo_set) + 'OpticalFlowIntoPast_' + idx_tgt + '_R.pfm'
			disp_c_l1_future = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_future/',stereo_set, idx_src) + '.pfm'
			#disp_c_r1_future = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_future/', stereo_set, idx_src) + '.pfm'
			disp_c_l2_past = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_past/', stereo_set, idx_tgt) + '.pfm'
			#disp_c_r2_past = os.path.join(images_root, disp_change_folder, dir_setname, freq_foldname, 'into_past/', stereo_set, idx_tgt) + '.pfm'


			if not (os.path.isfile(name_l1) and os.path.isfile(name_l2)):
				raise ValueError("Image file '%s' not found!", name_l1)

			if not (os.path.isfile(disp_l1) and os.path.isfile(disp_l1)):
				raise ValueError("Disp file '%s' not found!", disp_r1)

			if not (os.path.isfile(disp_c_l1_future) and os.path.isfile(disp_c_l2_past)):
				raise ValueError("Disp Change file '%s' not found!", disp_c_l1_future)

			if not (os.path.isfile(flow_l1_future) and os.path.isfile(flow_l2_past)):
				ValueError("Flow file '%s' not found!", flow_l1_future)

			self._image_list.append([name_l1, name_l2])
			self._future_list.append([disp_l1, disp_c_l1_future, flow_l1_future])
			self._past_list.append([disp_l2, disp_c_l2_past, flow_l2_past])

		#print("dataset Length:", len(self._image_list))
		if num_examples > 0:
			self._image_list = self._image_list[:num_examples]
			self._future_list = self._future_list[:num_examples]
			self._past_list = self._past_list[:num_examples]

		self._size = len(self._image_list)

		## loading calibration matrix
		self.intrinsic = intrinsic 

		self._to_tensor = vision_transforms.Compose([
			vision_transforms.ToPILImage(),
			vision_transforms.transforms.ToTensor()
		])

	def __getitem__(self, index):
		index = index % self._size

		# read images and flow
		# im_l1, im_l2, im_r1, im_r2
		#img_list_np = [Image.open(img).convert('RGB') for img in self._image_list[index]]
		img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]
		# displ0, dispr0, dispC_l0_future, dispC_r0_future, flowl0_future, flowr0_future
		future_list_np = [readPFM(img)[0] for img in self._future_list[index]]
		# displ1, dispr1, dispC_l1_past, dispC_r1_past, flowl1_past, flowr1_past
		past_list_np = [readPFM(img)[0] for img in self._past_list[index]]
		h_orig, w_orig, _ = img_list_np[0].shape
		crop_size = [h_orig, w_orig]


		im0 = img_list_np[0]
		im1 = img_list_np[1]
		disp0, dispC0, flow0 = future_list_np
		disp1, dispC1, flow1 = past_list_np 
		disp0_future = np.abs(disp0)
		disp1_future = np.abs(disp0 + dispC0)
		disp1_past = np.abs(disp1)
		disp0_past = np.abs(disp1 + dispC1)
		# disp0_future = cv2.resize(disp0_future,(intPreprocessedHeight,intPreprocessedWidth))
		# disp1_future = cv2.resize(disp1_future,(intPreprocessedHeight,intPreprocessedWidth))
		# disp1_past = cv2.resize(disp1_past,(intPreprocessedHeight,intPreprocessedWidth))
		# disp0_past = cv2.resize(disp0_past,(intPreprocessedHeight,intPreprocessedWidth))
		flow0[:,:,2] = 1
		flow1[:,:,2] = 1
		# flow0 = cv2.resize(flow0,(intPreprocessedHeight,intPreprocessedWidth))
		# flow1 = cv2.resize(flow1,(intPreprocessedHeight,intPreprocessedWidth))

		bl = 1
		fl = self.intrinsic[0,0]
		cx = self.intrinsic[0,2]
		cy = self.intrinsic[1,2]
		# into future 
		img0_crop_f, img1_crop_f, flow0_f, imgAux_f, occp0_f = generate_gt_expansion(im0, im1, self._datashape, flow0,disp0_future, disp1_future, bl, fl, cx, cy,
																													'%s/iter_counts.txt'%(self._args.save), order=1, prob=1)
		img0_crop_b, img1_crop_b, flow0_b, imgAux_b, occp0_b = generate_gt_expansion(im1, im0, self._datashape, flow1,disp1_past, disp0_past, bl, fl, cx, cy,
																													'%s/iter_counts.txt'%(self._args.save), order=1, prob=1)
		
		#print("image0_crop_f shape:", img0_crop_f.shape)
		# future set
		img0_crop_f = numpy2torch(img0_crop_f)
		img1_crop_f = numpy2torch(img1_crop_f)
		flow0_f = numpy2torch(flow0_f)
		imgAux_f = numpy2torch(imgAux_f)
		#intr_f = torch.from_numpy(np.asarray(intr_f).copy()).float()
		#imgAug0_f = torch.from_numpy(imgAug0_f.copy()).float()
		#imgAug1_f = torch.from_numpy(imgAug1_f.copy()).float()
		occp0_f = torch.from_numpy(occp0_f.copy()).float()
		# past set
		img0_crop_b = numpy2torch(img0_crop_b)
		img1_crop_b = numpy2torch(img1_crop_b)
		flow0_b = numpy2torch(flow0_b)
		imgAux_b = numpy2torch(imgAux_b)
		#intr_b = torch.from_numpy(np.asarray(intr_b).copy()).float()
		#imgAug0_b = torch.from_numpy(imgAug0_b.copy()).float()
		#imgAug1_b = torch.from_numpy(imgAug1_b.copy()).float()
		occp0_b = torch.from_numpy(occp0_b.copy()).float()

		# example filename
		k_l1 = torch.from_numpy(self.intrinsic).float()
		k_r1 = torch.from_numpy(self.intrinsic).float()
		#print("k_l1 shape:", k_l1.shape)
		
		#img_list_tensor = [self._to_tensor(img) for img in img_list_tmp]
		# input size
		input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()
	   
		common_dict = {
			"input_size": input_im_size
		}

		# random flip
		# if self._flip_augmentations is True and torch.rand(1) > 0.5:
		# 	_, _, ww = img0_crop_f.size()
		# 	#print(im_l1.size())
		# 	img0_crop_f_flip = torch.flip(img0_crop_f, dims=[2])
		# 	img1_crop_f_flip = torch.flip(img1_crop_f, dims=[2])
		# 	flow0_f_flip = torch.flip(flow0_f, dims=[2])
		# 	imgAux_f_flip = torch.flip(imgAux_f, dims=[2])
		# 	imgAug0_f_flip = torch.flip(imgAug0_f, dims=[2])
		# 	imgAug1_f_flip = torch.flip(imgAug1_f, dims=[2])
		# 	occp0_f_flip = torch.flip(occp0_f, dims=[2])

		# 	img0_crop_b_flip = torch.flip(img0_crop_b, dims=[2])
		# 	img1_crop_b_flip = torch.flip(img1_crop_b, dims=[2])
		# 	flow0_b_flip = torch.flip(flow0_b, dims=[2])
		# 	imgAux_b_flip = torch.flip(imgAux_b, dims=[2])
		# 	imgAug0_b_flip = torch.flip(imgAug0_b, dims=[2])
		# 	imgAug1_b_flip = torch.flip(imgAug1_b, dims=[2])
		# 	occp0_b_flip = torch.flip(occp0_b, dims=[2])

		# 	intr_f[0, 2] = ww - intr_f[0, 2]
		# 	intr_b[0, 2] = ww - intr_b[0, 2]

		# 	example_dict = {
		# 		"im0_f": img0_crop_f_flip,
		# 		"im1_f": img1_crop_f_flip,
		# 		"flow_f": flow0_f_flip,
		# 		"imgAux_f": imgAux_f_flip,
		# 		"intr_f": intr_f, 
		# 		"im0Aug_f": imgAug0_f_flip,
		# 		"im1Aug_f": imgAug1_f_flip,
		# 		"occp0_f": occp0_f_flip,
		# 		"im0_b": img0_crop_b_flip,
		# 		"im1_b": img1_crop_b_flip,
		# 		"flow_b": flow0_b_flip,
		# 		"imgAux_b": imgAux_b_flip,
		# 		"intr_b": intr_b, 
		# 		"im0Aug_b": imgAug0_b_flip,
		# 		"im1Aug_b": imgAug1_b_flip,
		# 		"occp0_b": occp0_b_flip,
		# 	}
		# 	example_dict.update(common_dict)

		# else:
		example_dict = {
				"im0_f": img0_crop_f,
				"im1_f": img1_crop_f,
				"flow_f": flow0_f,
				"imgAux_f": imgAux_f,
				#"intr_f": intr_f, 
				#"im0Aug_f": imgAug0_f,
				#"im1Aug_f": imgAug1_f,
				"occp0_f": occp0_f,
				"im0_b": img0_crop_b,
				"im1_b": img1_crop_b,
				"flow_b": flow0_b,
				"imgAux_b": imgAux_b,
				#"intr_b": intr_b, 
				#"im0Aug_b": imgAug0_b,
				#"im1Aug_b": imgAug1_b,
				"occp0_b": occp0_b,
			}
		example_dict.update(common_dict)

		return example_dict

	def __len__(self):
		return self._size

class Synth_Driving_15mm_slow(ConcatDataset):
	def __init__(self,
				 args,
				 root,
				 datashape=[256,704],
				 num_examples=-1):


		self.dataset_fl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=True,
				 					forward_set=True,
				 					left_cam=True)
		self.dataset_fr = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=True,
				 					forward_set=True,
				 					left_cam=False)
		self.dataset_bl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=True,
				 					forward_set=False,
				 					left_cam=True)
		self.dataset_br = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=True,
				 					forward_set=False,
				 					left_cam=False)
		#print("fl dataset Length", len(self.dataset_fl))
		super(Synth_Driving_15mm_slow, self).__init__(
			datasets=[self.dataset_fl, self.dataset_fr, self.dataset_bl, self.dataset_br])

class Synth_Driving_35mm_slow(ConcatDataset):
	def __init__(self,
				 args,
				 root,
				 datashape=[256,704],
				 num_examples=-1):


		self.dataset_fl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=False,
				 					forward_set=True,
				 					left_cam=True)
		self.dataset_fr = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=False,
				 					forward_set=True,
				 					left_cam=False)
		self.dataset_bl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=False,
				 					forward_set=False,
				 					left_cam=True)
		self.dataset_br = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=False,
				 					focal_15mm=False,
				 					forward_set=False,
				 					left_cam=False)
		super(Synth_Driving_35mm_slow, self).__init__(
			datasets=[self.dataset_fl, self.dataset_fr, self.dataset_bl, self.dataset_br])
		

class Synth_Driving_15mm_fast(ConcatDataset):
	def __init__(self,
				 args,
				 root,
				 datashape=[256,704],
				 num_examples=-1):


		self.dataset_fl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=True,
				 					forward_set=True,
				 					left_cam=True)
		self.dataset_fr = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=True,
				 					forward_set=True,
				 					left_cam=False)
		self.dataset_bl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=True,
				 					forward_set=False,
				 					left_cam=True)
		self.dataset_br = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=True,
				 					forward_set=False,
				 					left_cam=False)
		super(Synth_Driving_15mm_fast, self).__init__(
			datasets=[self.dataset_fl, self.dataset_fr, self.dataset_bl, self.dataset_br])


class Synth_Driving_35mm_fast(ConcatDataset):
	def __init__(self,
				 args,
				 root,
				 datashape=[256,704],
				 num_examples=-1):


		self.dataset_fl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=False,
				 					forward_set=True,
				 					left_cam=True)
		self.dataset_fr = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=False,
				 					forward_set=True,
				 					left_cam=False)
		self.dataset_bl = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=False,
				 					forward_set=False,
				 					left_cam=True)
		self.dataset_br = Synth_Driving(args,
				 					images_root=root,
				 					datashape=datashape,
				 					num_examples=num_examples,
				 					fast_set=True,
				 					focal_15mm=False,
				 					forward_set=False,
				 					left_cam=False)
		super(Synth_Driving_35mm_fast, self).__init__(
			datasets=[self.dataset_fl, self.dataset_fr, self.dataset_bl, self.dataset_br])

class Synth_Driving_Train(ConcatDataset):  
	def __init__(self, args, root, num_examples=-1):        

		self.dataset1 = Synth_Driving_15mm_slow(
			args, 
			root=root,
			num_examples=num_examples//2)

		self.dataset2 = Synth_Driving_35mm_slow(
			args, 
			root=root,
			num_examples=num_examples//2)

		self.dataset3 = Synth_Driving_15mm_fast(
			args, 
			root=root,
			num_examples=num_examples//2)

		super(Synth_Driving_Train, self).__init__(
			datasets=[self.dataset1, self.dataset2, self.dataset3])

class Synth_Driving_Val(Synth_Driving_35mm_fast):  
	def __init__(self, args, root, num_examples=-1):        
		super(Synth_Driving_Val, self).__init__(
			args, 
			root=root,
			num_examples=num_examples//2)

class Synth_Driving_Full(ConcatDataset):  
	def __init__(self, args, root, num_examples=-1):        

		self.dataset1 = Synth_Driving_Train(
			args, 
			root=root,
			num_examples=num_examples//2)

		self.dataset2 = Synth_Driving_Val(
			args, 
			root=root,
			num_examples=num_examples//2)

		super(Synth_Driving_Full, self).__init__(
			datasets=[self.dataset1, self.dataset2])