rom __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, read_png_flow, read_png_disp, numpy2torch
from .common import kitti_crop_image_list, kitti_adjust_intrinsic, intrinsic_scale, get_date_from_width
from .common import list_flatten, threed_warp, readPFM, generate_gt_expansion

import torch.nn.functional as tf
import glob
from torch.utils.data.dataset import ConcatDataset


class Synth_Driving(data.Dataset):
	def __init__(self,
				 args,
				 images_root=None,
				 flip_augmentations=True,
				 num_examples=-1,
				 fast_set=False,
				 focal_15mm=True):

		self._args = args
		self._seq_len = 1
		self._flip_augmentations = flip_augmentations		

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

		self._image_list = []
		self._disp_list = []
		self._disp_c_list = []
		self._flow_list = []
		image_folder = 'frames_cleanpass/' + subfolder
		disp_folder = 'disparity/' + subfolder
		disp_change_folder = 'disparity_change/' + subfolder
		flow_folder = 'optical_flow/' + subfolder
		#print(images_root)
		
		for idx in (total_img_num - 1):
			idx_src = '%.4d' % (int(idx))
			idx_tgt = '%.4d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, image_folder, 'scene_forwards/', freq_foldname, 'left/', idx_src) + '.png'
			name_l2 = os.path.join(images_root, image_folder, 'scene_forwards/', freq_foldname, 'left/', idx_tgt) + '.png'
			name_r1 = os.path.join(images_root, image_folder, 'scene_forwards/', freq_foldname, 'right/', idx_src) + '.png'
			name_r2 = os.path.join(images_root, image_folder, 'scene_forwards/', freq_foldname, 'right/', idx_tgt) + '.png'
			disp_l1 = os.path.join(images_root, disp_folder, 'scene_forwards/', freq_foldname, 'left/', idx_src) + '.pfm'
			disp_l2 = os.path.join(images_root, disp_folder, 'scene_forwards/', freq_foldname, 'left/', idx_tgt) + '.pfm'
			disp_r1 = os.path.join(images_root, disp_folder, 'scene_forwards/', freq_foldname, 'right/', idx_src) + '.pfm'
			disp_r2 = os.path.join(images_root, disp_folder, 'scene_forwards/', freq_foldname, 'right/', idx_tgt) + '.pfm'
			opt_flow_l1_future = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_future/', 'left/') + 'OpticalFlowIntoFuture_' + idx_src + '_L.pfm'
			opt_flow_r1_future = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_future/', 'right/') + 'OpticalFlowIntoFuture_' + idx_src + '_R.pfm'
			opt_flow_l1_past = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_past/', 'left/') + 'OpticalFlowIntoPast_' + idx_src + '_L.pfm'
			opt_flow_r1_past = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_past/', 'right/') + 'OpticalFlowIntoPast_' + idx_src + '_R.pfm'
			disp_c_l1_future = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_future/', 'left/', idx_src) + '.pfm'
			disp_c_r1_future = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_future/', 'right/', idx_src) + '.pfm'
			disp_c_l1_past = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_past/', 'left/', idx_src) + '.pfm'
			disp_c_r1_past = os.path.join(images_root, disp_change_folder, 'scene_forwards/', freq_foldname, 'into_past/', 'right/', idx_src) + '.pfm'


			if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2):
				self._image_list.append([name_l1, name_l2, name_r1, name_r2])
			else:
				raise ValueError("Image file '%s' not found!", name_l1)

			if os.path.isfile(disp_l1) and os.path.isfile(disp_r1) and os.path.isfile(disp_l2) and os.path.isfile(disp_r2):
				self._disp_list.append([disp_l1, disp_r1, disp_l2, disp_r2])
			else:
				raise ValueError("Disp file '%s' not found!", disp_r1)

			if os.path.isfile(disp_c_l1_future) and os.path.isfile(disp_c_r1_future) and os.path.isfile(disp_c_l1_past) and os.path.isfile(disp_c_r1_past):
				self._disp_c_list.append([disp_c_l1_future, disp_c_r1_future, disp_c_l1_past, disp_c_r1_past])
			else:
				raise ValueError("Disp Change file '%s' not found!", disp_c_l1_future)

			if os.path.isfile(opt_flow_l1_future) and os.path.isfile(opt_flow_r1_future) and os.path.isfile(opt_flow_l1_past) and os.path.isfile(opt_flow_r1_past):
				self._disp_c_list.append([opt_flow_l1_future, opt_flow_r1_future, opt_flow_l1_past, opt_flow_r1_past])
			else:
				raise ValueError("Flow file '%s' not found!", opt_flow_l1_future)

		for idx in (total_img_num - 1):
			idx_src = '%.4d' % (int(idx))
			idx_tgt = '%.4d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, image_folder, 'scene_backwards/', freq_foldname, 'left/', idx_src) + '.png'
			name_l2 = os.path.join(images_root, image_folder, 'scene_backwards/', freq_foldname, 'left/', idx_tgt) + '.png'
			name_r1 = os.path.join(images_root, image_folder, 'scene_backwards/', freq_foldname, 'right/', idx_src) + '.png'
			name_r2 = os.path.join(images_root, image_folder, 'scene_backwards/', freq_foldname, 'right/', idx_tgt) + '.png'
			disp_l1 = os.path.join(images_root, disp_folder, 'scene_backwards/', freq_foldname, 'left/', idx_src) + '.pfm'
			disp_l2 = os.path.join(images_root, disp_folder, 'scene_backwards/', freq_foldname, 'left/', idx_tgt) + '.pfm'
			disp_r1 = os.path.join(images_root, disp_folder, 'scene_backwards/', freq_foldname, 'right/', idx_src) + '.pfm'
			disp_r2 = os.path.join(images_root, disp_folder, 'scene_backwards/', freq_foldname, 'right/', idx_tgt) + '.pfm'
			opt_flow_l1_future = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_future/', 'left/') + 'OpticalFlowIntoFuture_' + idx_src + '_L.pfm'
			opt_flow_r1_future = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_future/', 'right/') + 'OpticalFlowIntoFuture_' + idx_src + '_R.pfm'
			opt_flow_l1_past = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_past/', 'left/') + 'OpticalFlowIntoPast_' + idx_src + '_L.pfm'
			opt_flow_r1_past = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_past/', 'right/') + 'OpticalFlowIntoPast_' + idx_src + '_R.pfm'
			disp_c_l1_future = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_future/', 'left/', idx_src) + '.pfm'
			disp_c_r1_future = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_future/', 'right/', idx_src) + '.pfm'
			disp_c_l1_past = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_past/', 'left/', idx_src) + '.pfm'
			disp_c_r1_past = os.path.join(images_root, disp_change_folder, 'scene_backwards/', freq_foldname, 'into_past/', 'right/', idx_src) + '.pfm'


			if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2):
				self._image_list.append([name_l1, name_l2, name_r1, name_r2])
			else:
				raise ValueError("Image file '%s' not found!", name_l1)

			if os.path.isfile(disp_l1) and os.path.isfile(disp_r1) and os.path.isfile(disp_l2) and os.path.isfile(disp_r2):
				self._disp_list.append([disp_l1, disp_r1, disp_l2, disp_r2])
			else:
				raise ValueError("Disp file '%s' not found!", disp_r1)

			if os.path.isfile(disp_c_l1_future) and os.path.isfile(disp_c_r1_future) and os.path.isfile(disp_c_l1_past) and os.path.isfile(disp_c_r1_past):
				self._disp_c_list.append([disp_c_l1_future, disp_c_r1_future, disp_c_l1_past, disp_c_r1_past])
			else:
				raise ValueError("Disp Change file '%s' not found!", disp_c_l1_future)

			if os.path.isfile(opt_flow_l1_future) and os.path.isfile(opt_flow_r1_future) and os.path.isfile(opt_flow_l1_past) and os.path.isfile(opt_flow_r1_past):
				self._flow_list.append([opt_flow_l1_future, opt_flow_r1_future, opt_flow_l1_past, opt_flow_r1_past])
			else:
				raise ValueError("Flow file '%s' not found!", opt_flow_l1_future)

		if num_examples > 0:
			self._image_list = self._image_list[:num_examples]

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
		img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]
		disp_list_np = [readPFM(img) for img in self._disp_list[index]]
		flow_list_np = [readPFM(img) for img in self._flow_list[index]]
		dispC_list_np = [readPFM(img) for img in self._disp_c_list[index]]

		# disp set as positive?
		for ii in len(flow_list_np):
			flow_list_np[ii][np.isnan(flow)] = 1e6 

		depth_l1, gt_normal_l1, flow3d_l1 = generate_gt_expansion(flow_list_np[0], disp_list_np[0], dispC_list_np[0], self.intrinsic)
		depth_r1, gt_normal_r1, flow3d_r1 = generate_gt_expansion(flow_list_np[1], disp_list_np[1], dispC_list_np[1], self.intrinsic)
		depth_l1 = numpy2torch(depth_l1)
		gt_normal_l1 = numpy2torch(gt_normal_l1)
		flow3d_l1 = numpy2torch(flow3d_l1)
		depth_r1 = numpy2torch(depth_r1)
		gt_normal_r1 = numpy2torch(gt_normal_r1)
		flow3d_r1 = numpy2torch(flow3d_r1)


		# example filename
		k_l1 = torch.from_numpy(self.intrinsic).float()
		k_r1 = torch.from_numpy(self.intrinsic).float()
		#print("k_l1 shape:", k_l1.shape)
		
		# input size
		h_orig, w_orig, _ = img_list_np[0].shape
		input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()
		
		# to tensors
		img_list_tensor = [self._to_tensor(img) for img in img_list_np]
		disp_list_tensor = [numpy2torch(img) for img in disp_list_np]
		flow_list_tensor = [numpy2torch(img) for img in flow_list_np]
		dispC_list_tensor = [numpy2torch(img) for img in dispC_list_np]
		depth0_list_tensor = [numpy2torch(img) for img in depth0_list_np]
		gt_normal_list_tensor = [numpy2torch(img) for img in gt_normal_list_np]
		flow3d_list_tensor = [numpy2torch(img) for img in flow3d_list_np]

		
		im_l1 = img_list_tensor[0]
		im_l2 = img_list_tensor[1]
		im_r1 = img_list_tensor[2]
		im_r2 = img_list_tensor[3]

		disp_l1 = disp_list_tensor[0]
		disp_r1 = disp_list_tensor[1]
		disp_l2 = disp_list_tensor[2]
		disp_r2 = disp_list_tensor[3]

		flow_l1_future = flow_list_tensor[0]
		flow_r1_future = flow_list_tensor[1]
		flow_l1_past = flow_list_tensor[2]
		flow_r1_past = flow_list_tensor[3]

		dispC_l1_future = dispC_list_tensor[0]
		dispC_r1_future = dispC_list_tensor[1]
		dispC_l2_past = dispC_list_tensor[2]
		dispC_r2_past = dispC_list_tensor[3]

		dispC_l1_future = dispC_list_tensor[0]
		dispC_r1_future = dispC_list_tensor[1]
		dispC_l2_past = dispC_list_tensor[2]
		dispC_r2_past = dispC_list_tensor[3]

		dispC_l1_future = dispC_list_tensor[0]
		dispC_r1_future = dispC_list_tensor[1]
		dispC_l2_past = dispC_list_tensor[2]
		dispC_r2_past = dispC_list_tensor[3]

		dispC_l1_future = dispC_list_tensor[0]
		dispC_r1_future = dispC_list_tensor[1]
		dispC_l2_past = dispC_list_tensor[2]
		dispC_r2_past = dispC_list_tensor[3]
	   
		common_dict = {
			"input_size": input_im_size
		}

		# random flip
		if self._flip_augmentations is True and torch.rand(1) > 0.5:
			_, _, ww = im_l1.size()
			#print(im_l1.size())
			im_l1_flip = torch.flip(im_l1, dims=[2])
			im_l2_flip = torch.flip(im_l2, dims=[2])
			im_r1_flip = torch.flip(im_r1, dims=[2])
			im_r2_flip = torch.flip(im_r2, dims=[2])

			disp_l1_flip = torch.flip(disp_l1, dims=[2])
			disp_r1_flip = torch.flip(disp_r1, dims=[2])
			disp_l2_flip = torch.flip(disp_l2, dims=[2])
			disp_r2_flip = torch.flip(disp_r2, dims=[2])


			flow_l1_future_flip = torch.flip(flow_l1_future, dims=[2])
			flow_r1_future_flip = torch.flip(flow_r1_future, dims=[2])
			flow_l1_past_flip = torch.flip(flow_l1_past, dims=[2])
			flow_r1_past_flip = torch.flip(flow_r1_past, dims=[2])

			dispC_l1_future_flip = torch.flip(dispC_l1_future, dims=[2])
			dispC_r1_future_flip = torch.flip(dispC_r1_future, dims=[2])
			dispC_l1_past_flip = torch.flip(dispC_l1_past, dims=[2])
			dispC_r1_past_flip = torch.flip(dispC_r1_past, dims=[2])

			depth_l1_flip = torch.flip(depth_l1, dims=[2])
			gt_normal_l1_flip = torch.flip(gt_normal_l1, dims=[2])
			flow3d_l1_flip = torch.flip(flow3d_l1, dims=[2])

			depth_r1_flip = torch.flip(depth_r1, dims=[2])
			gt_normal_r1_flip = torch.flip(gt_normal_r1, dims=[2])
			flow3d_r1_flip = torch.flip(flow3d_r1, dims=[2])

			k_l1[0, 2] = ww - k_l1[0, 2]
			k_r1[0, 2] = ww - k_r1[0, 2]

			example_dict = {
				"input_l1": im_r1_flip,
				"input_r1": im_l1_flip,
				"input_l2": im_r2_flip,
				"input_r2": im_l2_flip,                
				"input_k_l1": k_r1,
				"input_k_r1": k_l1,
				"input_k_l2": k_r1,
				"input_k_r2": k_l1,
				"input_disp_l1": disp_l1_flip,
				"input_disp_r1": disp_r1_flip,
				"input_disp_l2": disp_l2_flip,
				"input_disp_r2": disp_r2_flip,
				"input_flow_l1_future": flow_l1_future_flip,
				"input_flow_r1_future": flow_r1_future_flip,
				"input_flow_l1_past": flow_l1_past_flip,
				"input_flow_r1_past": flow_r1_past_flip,
				"input_dispC_l1_future": dispC_l1_future_flip,
				"input_dispC_r1_future": dispC_r1_future_flip,
				"input_dispC_l1_past": dispC_l1_past_flip,
				"input_dispC_r1_past": dispC_r1_past_flip,
				"input_depth_l1": depth_l1_flip,
				"input_gt_normal_l1": gt_normal_l1_flip,
				"input_flow3d_l1": flow3d_l1_flip,
				"input_depth_r1": depth_r1_flip,
				"input_gt_normal_r1": gt_normal_r1_flip,
				"input_flow3d_r1": flow3d_r1_flip,
			}
			example_dict.update(common_dict)

		else:
			example_dict = {
				"input_l1": im_l1,
				"input_r1": im_r1,
				"input_l2": im_l2,
				"input_r2": im_r2,
				"input_k_l1": k_l1,
				"input_k_r1": k_r1,
				"input_k_l2": k_l1,
				"input_k_r2": k_r1,
				"input_disp_l1": disp_l1,
				"input_disp_r1": disp_r1,
				"input_disp_l2": disp_l2,
				"input_disp_r2": disp_r2,
				"input_flow_l1_future": flow_l1_future,
				"input_flow_r1_future": flow_r1_future,
				"input_flow_l1_past": flow_l1_past,
				"input_flow_r1_past": flow_r1_past,
				"input_dispC_l1_future": dispC_l1_future,
				"input_dispC_r1_future": dispC_r1_future,
				"input_dispC_l1_past": dispC_l1_past,
				"input_dispC_r1_past": dispC_r1_past,
				"input_depth_l1": depth_l1,
				"input_gt_normal_l1": gt_normal_l1,
				"input_flow3d_l1": flow3d_l1,
				"input_depth_r1": depth_r1,
				"input_gt_normal_r1": gt_normal_r1,
				"input_flow3d_r1": flow3d_r1,
			}
			example_dict.update(common_dict)

		return example_dict

	def __len__(self):
		return self._size

class Synth_Driving_15mm_slow(Synth_Driving):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 num_examples=-1):
		super(Synth_Driving_15mm_slow, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			num_examples=num_examples,
			fast_set=False,
			focal_15mm=True)

class Synth_Driving_35mm_slow(Synth_Driving):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 num_examples=-1):
		super(Synth_Driving_35mm_slow, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			num_examples=num_examples,
			fast_set=False,
			focal_15mm=False)


class Synth_Driving_15mm_fast(Synth_Driving):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 num_examples=-1):
		super(Synth_Driving_15mm_fast, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			num_examples=num_examples,
			fast_set=True,
			focal_15mm=True)

class Synth_Driving_35mm_fast(Synth_Driving):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 num_examples=-1):
		super(Synth_Driving_35mm_fast, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			num_examples=num_examples,
			fast_set=True,
			focal_15mm=False)

class Synth_Driving_Train(ConcatDataset):  
	def __init__(self, args, root, num_examples=-1):        

		self.dataset1 = Synth_Driving_15mm_slow(
			args, 
			root=root, 
			preprocessing_crop=True,
			num_examples=num_examples//2)

		self.dataset2 = Synth_Driving_35mm_slow(
			args, 
			root=root,
			flip_augmentations=True,
			num_examples=num_examples//2)

		super(Synth_Driving_Train, self).__init__(
			datasets=[self.dataset1, self.dataset2])

class Synth_Driving_Val(ConcatDataset):  
	def __init__(self, args, root, num_examples=-1):        

		self.dataset1 = Synth_Driving_15mm_fast(
			args, 
			root=root, 
			preprocessing_crop=True,
			num_examples=num_examples//2)

		self.dataset2 = Synth_Driving_35mm_fast(
			args, 
			root=root,
			flip_augmentations=True,
			num_examples=num_examples//2)

		super(Synth_Driving_Val, self).__init__(
			datasets=[self.dataset1, self.dataset2])