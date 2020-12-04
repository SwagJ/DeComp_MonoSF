from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, read_png_flow, read_png_disp, numpy2torch
from .common import kitti_crop_image_list, kitti_adjust_intrinsic, intrinsic_scale, get_date_from_width
from .common import list_flatten

import torch.nn.functional as tf


class KITTI_Raw(data.Dataset):
	def __init__(self,
				 args,
				 images_root=None,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1,
				 index_file=None):

		self._args = args
		self._seq_len = 1
		self._flip_augmentations = flip_augmentations
		self._preprocessing_crop = preprocessing_crop
		self._crop_size = crop_size

		path_dir = os.path.dirname(os.path.realpath(__file__))
		path_index_file = os.path.join(path_dir, index_file)

		if not os.path.exists(path_index_file):
			raise ValueError("Index File '%s' not found!", path_index_file)
		index_file = open(path_index_file, 'r')

		## loading image -----------------------------------
		if not os.path.isdir(images_root):
			raise ValueError("Image directory '%s' not found!")

		filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
		self._image_list = []
		view1 = 'image_02/data'
		view2 = 'image_03/data'
		ext = '.png'
		
		for item in filename_list:
			date = item[0][:10]
			scene = item[0]
			idx_src = item[1]
			idx_tgt = '%.10d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
			name_l2 = os.path.join(images_root, date, scene, view1, idx_tgt) + ext
			name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
			name_r2 = os.path.join(images_root, date, scene, view2, idx_tgt) + ext

			if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2):
				self._image_list.append([name_l1, name_l2, name_r1, name_r2])

		if num_examples > 0:
			self._image_list = self._image_list[:num_examples]

		self._size = len(self._image_list)

		## loading calibration matrix
		self.intrinsic_dict_l = {}
		self.intrinsic_dict_r = {}        
		self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

		self._to_tensor = vision_transforms.Compose([
			vision_transforms.ToPILImage(),
			vision_transforms.transforms.ToTensor()
		])

	def __getitem__(self, index):
		index = index % self._size

		# read images and flow
		# im_l1, im_l2, im_r1, im_r2
		img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

		# example filename
		im_l1_filename = self._image_list[index][0]
		basename = os.path.basename(im_l1_filename)[:6]
		dirname = os.path.dirname(im_l1_filename)[-51:]
		datename = dirname[:10]
		k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
		k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
		
		# input size
		h_orig, w_orig, _ = img_list_np[0].shape
		input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

		# cropping 
		if self._preprocessing_crop:

			# get starting positions
			crop_height = self._crop_size[0]
			crop_width = self._crop_size[1]
			x = np.random.uniform(0, w_orig - crop_width + 1)
			y = np.random.uniform(0, h_orig - crop_height + 1)
			crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

			# cropping images and adjust intrinsic accordingly
			img_list_np = kitti_crop_image_list(img_list_np, crop_info)
			k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
		
		# to tensors
		img_list_tensor = [self._to_tensor(img) for img in img_list_np]
		
		im_l1 = img_list_tensor[0]
		im_l2 = img_list_tensor[1]
		im_r1 = img_list_tensor[2]
		im_r2 = img_list_tensor[3]
	   
		common_dict = {
			"index": index,
			"basename": basename,
			"datename": datename,
			"input_size": input_im_size
		}

		# random flip
		if self._flip_augmentations is True and torch.rand(1) > 0.5:
			_, _, ww = im_l1.size()
			im_l1_flip = torch.flip(im_l1, dims=[2])
			im_l2_flip = torch.flip(im_l2, dims=[2])
			im_r1_flip = torch.flip(im_r1, dims=[2])
			im_r2_flip = torch.flip(im_r2, dims=[2])

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
			}
			example_dict.update(common_dict)

		return example_dict

	def __len__(self):
		return self._size



class KITTI_Raw_KittiSplit_Train(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_KittiSplit_Train, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_train.txt")


class KITTI_Raw_KittiSplit_Valid(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=False,
				 preprocessing_crop=False,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_KittiSplit_Valid, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_valid.txt")


class KITTI_Raw_KittiSplit_Full(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_KittiSplit_Full, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_full.txt")


class KITTI_Raw_EigenSplit_Train(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_EigenSplit_Train, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/eigen_train.txt")


class KITTI_Raw_EigenSplit_Valid(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=False,
				 preprocessing_crop=False,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_EigenSplit_Valid, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/eigen_valid.txt")


class KITTI_Raw_EigenSplit_Full(KITTI_Raw):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_EigenSplit_Full, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/eigen_full.txt")






### KITTI Raw Dataset with Depth Loaded. 
class KITTI_Raw_Depth(data.Dataset):
	def __init__(self,
				 args,
				 images_root=None,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1,
				 index_file=None):

		self._args = args
		self._seq_len = 1
		self._flip_augmentations = flip_augmentations
		self._preprocessing_crop = preprocessing_crop
		self._crop_size = crop_size

		path_dir = os.path.dirname(os.path.realpath(__file__))
		path_index_file = os.path.join(path_dir, index_file)

		if not os.path.exists(path_index_file):
			raise ValueError("Index File '%s' not found!", path_index_file)
		index_file = open(path_index_file, 'r')

		## loading image -----------------------------------
		if not os.path.isdir(images_root):
			raise ValueError("Image directory '%s' not found!")

		filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
		self._image_list = []
		self._depth_list = []
		view1 = 'image_02/data'
		view2 = 'image_03/data'
		depth1 = 'completed_depth/image_02'
		depth2 = 'completed_depth/image_03'
		ext = '.png'
		
		for item in filename_list:
			date = item[0][:10]
			scene = item[0]
			idx_src = item[1]
			idx_tgt = '%.10d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
			name_l2 = os.path.join(images_root, date, scene, view1, idx_tgt) + ext
			name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
			name_r2 = os.path.join(images_root, date, scene, view2, idx_tgt) + ext
			depth_l1 = os.path.join(images_root, date, scene, depth1, idx_src) + ext
			depth_l2 = os.path.join(images_root, date, scene, depth1, idx_tgt) + ext
			depth_r1 = os.path.join(images_root, date, scene, depth2, idx_src) + ext
			depth_r2 = os.path.join(images_root, date, scene, depth2, idx_tgt) + ext

			if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2) and os.path.isfile(depth_l1) and os.path.isfile(depth_l2) and os.path.isfile(depth_r1) and os.path.isfile(depth_r2):
				self._image_list.append([name_l1, name_l2, name_r1, name_r2])
				self._depth_list.append([depth_l1, depth_l2, depth_r1, depth_r2])

		if len(self._image_list) != len(self._depth_list):
			raise ValueError("Image and Depth Incosistency!!! Double Check")

		for idx, _ in enumerate(self._image_list):
			#print(self._image_list[idx])
			for sub_idx, _ in enumerate(self._image_list[idx]):
				if not os.path.isfile(self._image_list[idx][sub_idx]):
					raise ValueError("Image File not exist: %s", self._image_list[idx])
				if not os.path.isfile(self._depth_list[idx][sub_idx]):
					raise ValueError("Depth File not exist: %s", self._depth_list[idx])


		if num_examples > 0:
			self._image_list = self._image_list[:num_examples]
			self._depth_list = self._depth_list[:num_examples]

		self._size = len(self._image_list)

		## loading calibration matrix
		self.intrinsic_dict_l = {}
		self.intrinsic_dict_r = {}        
		self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

		self._to_tensor = vision_transforms.Compose([
			vision_transforms.ToPILImage(),
			vision_transforms.transforms.ToTensor()
		])

	def __getitem__(self, index):
		index = index % self._size

		# read images and flow
		# im_l1, im_l2, im_r1, im_r2
		img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

		disp_list_np = [read_png_disp(img) for img in self._depth_list[index]]
		disp_list_np = list_flatten(disp_list_np)

		# example filename
		im_l1_filename = self._image_list[index][0]
		basename = os.path.basename(im_l1_filename)[:6]
		dirname = os.path.dirname(im_l1_filename)[-51:]
		datename = dirname[:10]
		k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
		k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
		#print("Disp size:",disp_list_np[0].shape)
		#print("Image size:",img_list_np[0].shape)
		
		# input size
		h_orig, w_orig, _ = img_list_np[0].shape
		input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

		# cropping 
		if self._preprocessing_crop:

			# get starting positions
			crop_height = self._crop_size[0]
			crop_width = self._crop_size[1]
			x = np.random.uniform(0, w_orig - crop_width + 1)
			y = np.random.uniform(0, h_orig - crop_height + 1)
			crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

			# cropping images and adjust intrinsic accordingly
			img_list_np = kitti_crop_image_list(img_list_np, crop_info)
			disp_list_np = kitti_crop_image_list(disp_list_np, crop_info)
			k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
		
		# to tensors
		img_list_tensor = [self._to_tensor(img) for img in img_list_np]
		disp_list_tensor = [numpy2torch(img) for img in disp_list_np]

		for i in range(len(img_list_tensor)):
			disp_list_tensor[2*i] = self.interpolate2d_as(disp_list_tensor[2*i],img_list_tensor[i])
			disp_list_tensor[2*i+1] = self.interpolate2d_as(disp_list_tensor[2*i+1],img_list_tensor[i])

		
		im_l1 = img_list_tensor[0]
		im_l2 = img_list_tensor[1]
		im_r1 = img_list_tensor[2]
		im_r2 = img_list_tensor[3]

		disp_l1 = disp_list_tensor[0]
		disp_l2 = disp_list_tensor[2]
		disp_r1 = disp_list_tensor[4]
		disp_r2 = disp_list_tensor[6]

		disp_l1_mask = disp_list_tensor[1]
		disp_l2_mask = disp_list_tensor[3]
		disp_r1_mask = disp_list_tensor[5]
		disp_r2_mask = disp_list_tensor[7]

		#print("after cropping, Disp size:",disp_l1.shape)
		#print("after cropping, Image size:",im_l1.shape)
	   
		common_dict = {
			"index": index,
			"basename": basename,
			"datename": datename,
			"input_size": input_im_size
		}

		# random flip
		if self._flip_augmentations is True and torch.rand(1) > 0.5:
			_, _, ww = im_l1.size()
			im_l1_flip = torch.flip(im_l1, dims=[2])
			im_l2_flip = torch.flip(im_l2, dims=[2])
			im_r1_flip = torch.flip(im_r1, dims=[2])
			im_r2_flip = torch.flip(im_r2, dims=[2])

			disp_l1_flip = torch.flip(disp_l1, dims=[2])
			disp_l2_flip = torch.flip(disp_l2, dims=[2])
			disp_r1_flip = torch.flip(disp_r1, dims=[2])
			disp_r2_flip = torch.flip(disp_r2, dims=[2])

			disp_l1_mask_flip = torch.flip(disp_l1_mask, dims=[2])
			disp_l2_mask_flip = torch.flip(disp_l2_mask, dims=[2])
			disp_r1_mask_flip = torch.flip(disp_r1_mask, dims=[2])
			disp_r2_mask_flip = torch.flip(disp_r2_mask, dims=[2])

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
				"disp_l1": disp_l1_flip,
				"disp_l1_mask": disp_l1_mask_flip,
				"disp_l2": disp_l2_flip,
				"disp_l2_mask": disp_l2_mask_flip,
				"disp_r1": disp_r1_flip,
				"disp_r1_mask": disp_r1_mask_flip,
				"disp_r2": disp_r2_flip,
				"disp_r2_mask": disp_r2_mask_flip,
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
				"disp_l1": disp_l1,
				"disp_l1_mask": disp_l1_mask,
				"disp_l2": disp_l2,
				"disp_l2_mask": disp_l2_mask,
				"disp_r1": disp_r1,
				"disp_r1_mask": disp_r1_mask,
				"disp_r2": disp_r2,
				"disp_r2_mask": disp_r2_mask,
			}
			example_dict.update(common_dict)

		return example_dict

	def __len__(self):
		return self._size

	def interpolate2d_as(self, inputs, target_as, mode="bilinear"):
		_, _, h, w= target_as.unsqueeze(0).size()
		return tf.interpolate(inputs.unsqueeze(0), [h, w], mode=mode, align_corners=True).squeeze(0)


class KITTI_Raw_Depth_KittiSplit_Train(KITTI_Raw_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1,
				 index_file="index_txt/kitti_train.txt"):
		super(KITTI_Raw_Depth_KittiSplit_Train, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_train.txt")


class KITTI_Raw_Depth_KittiSplit_Valid(KITTI_Raw_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=False,
				 preprocessing_crop=False,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_Depth_KittiSplit_Valid, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_valid.txt")


class KITTI_Raw_Depth_KittiSplit_Full(KITTI_Raw_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_Depth_KittiSplit_Full, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_full.txt")



### KITTI Raw Dataset with Completed Depth Loaded for Sceneflow GT and extra augmentation. 
class KITTI_Raw_Completed_Depth(data.Dataset):
	def __init__(self,
				 args,
				 images_root=None,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1,
				 index_file=None):

		self._args = args
		self._seq_len = 1
		self._flip_augmentations = flip_augmentations
		self._preprocessing_crop = preprocessing_crop
		self._crop_size = crop_size

		path_dir = os.path.dirname(os.path.realpath(__file__))
		path_index_file = os.path.join(path_dir, index_file)

		if not os.path.exists(path_index_file):
			raise ValueError("Index File '%s' not found!", path_index_file)
		index_file = open(path_index_file, 'r')

		## loading image -----------------------------------
		if not os.path.isdir(images_root):
			raise ValueError("Image directory '%s' not found!")

		filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
		self._image_list = []
		self._depth_list = []
		self._fulldepth_list = []
		view1 = 'image_02/data'
		view2 = 'image_03/data'
		full_depth1 = 'completed_depth/image_02'
		full_depth2 = 'completed_depth/image_03'
		ext = '.png'
		
		for item in filename_list:
			date = item[0][:10]
			scene = item[0]
			idx_src = item[1]
			idx_tgt = '%.10d' % (int(idx_src) + 1)
			name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
			name_l2 = os.path.join(images_root, date, scene, view1, idx_tgt) + ext
			name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
			name_r2 = os.path.join(images_root, date, scene, view2, idx_tgt) + ext
			fulldepth_l1 = os.path.join(images_root, date, scene, full_depth1, idx_src) + ext
			fulldepth_l2 = os.path.join(images_root, date, scene, full_depth1, idx_tgt) + ext
			fulldepth_r1 = os.path.join(images_root, date, scene, full_depth2, idx_src) + ext
			fulldepth_r2 = os.path.join(images_root, date, scene, full_depth2, idx_tgt) + ext

			if (os.path.isfile(name_l1) and os.path.isfile(name_l2) and 
				os.path.isfile(name_r1) and os.path.isfile(name_r2) and 
				os.path.isfile(fulldepth_l1) and os.path.isfile(fulldepth_l2) and
				os.path.isfile(fulldepth_r1) and os.path.isfile(fulldepth_r2)):
				self._image_list.append([name_l1, name_l2, name_r1, name_r2])
				self._fulldepth_list.append([fulldepth_l1,fulldepth_l2,fulldepth_r1,fulldepth_r2]) 


		if len(self._image_list) != len(self._depth_list):
			raise ValueError("Image and Depth Incosistency!!! Double Check")

		for idx, _ in enumerate(self._image_list):
			#print(self._image_list[idx])
			for sub_idx, _ in enumerate(self._image_list[idx]):
				if not os.path.isfile(self._image_list[idx][sub_idx]):
					raise ValueError("Image File not exist: %s", self._image_list[idx])
				if not os.path.isfile(self._fulldepth_list[idx][sub_idx]):
					raise ValueError("Depth File not exist: %s", self._fulldepth_list[idx])


		if num_examples > 0:
			self._image_list = self._image_list[:num_examples]
			self._depth_list = self._depth_list[:num_examples]
			self._fulldepth_list = self._fulldepth_list[:num_examples]

		self._size = len(self._image_list)

		## loading calibration matrix
		self.intrinsic_dict_l = {}
		self.intrinsic_dict_r = {}        
		self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

		self._to_tensor = vision_transforms.Compose([
			vision_transforms.ToPILImage(),
			vision_transforms.transforms.ToTensor()
		])

	def __getitem__(self, index):
		index = index % self._size

		# read images and flow
		# im_l1, im_l2, im_r1, im_r2
		img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

		#disp_list_np = [read_png_disp(img) for img in self._depth_list[index]]
		#disp_list_np = list_flatten(disp_list_np)

		fulldisp_list_np = [read_png_disp(img) for img in self._fulldepth_list[index]]
		fulldisp_list_np = list_flatten(fulldisp_list_np)

		# example filename
		im_l1_filename = self._image_list[index][0]
		basename = os.path.basename(im_l1_filename)[:6]
		dirname = os.path.dirname(im_l1_filename)[-51:]
		datename = dirname[:10]
		k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
		k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
		
		# input size
		h_orig, w_orig, _ = img_list_np[0].shape
		input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

		# cropping 
		if self._preprocessing_crop:

			# get starting positions
			crop_height = self._crop_size[0]
			crop_width = self._crop_size[1]
			x = np.random.uniform(0, w_orig - crop_width + 1)
			y = np.random.uniform(0, h_orig - crop_height + 1)
			crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

			# cropping images and adjust intrinsic accordingly
			img_list_np = kitti_crop_image_list(img_list_np, crop_info)
			#disp_list_np = kitti_crop_image_list(disp_list_np, crop_info)
			fulldisp_list_np = kitti_crop_image_list(fulldisp_list_np, crop_info)
			k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
		
		# to tensors
		img_list_tensor = [self._to_tensor(img) for img in img_list_np]
		#disp_list_tensor = [numpy2torch(img) for img in disp_list_np]
		fulldisp_list_tensor = [numpy2torch(img) for img in fulldisp_list_np]
		
		im_l1 = img_list_tensor[0]
		im_l2 = img_list_tensor[1]
		im_r1 = img_list_tensor[2]
		im_r2 = img_list_tensor[3]

		fulldisp_l1 = fulldisp_list_tensor[0]
		fulldisp_l2 = fulldisp_list_tensor[2]
		fulldisp_r1 = fulldisp_list_tensor[4]
		fulldisp_r2 = fulldisp_list_tensor[6]

		#disp_l1_mask = disp_list_tensor[1]
		#disp_l2_mask = disp_list_tensor[3]
		#disp_r1_mask = disp_list_tensor[5]
		#disp_r2_mask = disp_list_tensor[7]
	   
		common_dict = {
			"index": index,
			"basename": basename,
			"datename": datename,
			"input_size": input_im_size
		}

		# random flip
		example_dict = {
				"input_l1": im_l1,
				"input_r1": im_r1,
				"input_l2": im_l2,
				"input_r2": im_r2,
				"input_k_l1": k_l1,
				"input_k_r1": k_r1,
				"input_k_l2": k_l1,
				"input_k_r2": k_r1,
				"fulldisp_l1": fulldisp_l1,
				"fulldisp_l2": fulldisp_l2,
				"fulldisp_r1": fulldisp_r1,
				"fulldisp_r2": fulldisp_r2,
		}
		example_dict.update(common_dict)

		return example_dict

	def __len__(self):
		return self._size

	def interpolate2d_as(self, inputs, target_as, mode="bilinear"):
		_, h, w = target_as.size()
		return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class KITTI_Raw_Completed_Depth_KittiSplit_Train(KITTI_Raw_Completed_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_Completed_Depth_KittiSplit_Train, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_train.txt")


class KITTI_Raw_Completed_Depth_KittiSplit_Valid(KITTI_Raw_Completed_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=False,
				 preprocessing_crop=False,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_Completed_Depth_KittiSplit_Valid, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_valid.txt")


class KITTI_Raw_Completed_Depth_KittiSplit_Full(KITTI_Raw_Completed_Depth):
	def __init__(self,
				 args,
				 root,
				 flip_augmentations=True,
				 preprocessing_crop=True,
				 crop_size=[370, 1224],
				 num_examples=-1):
		super(KITTI_Raw_Completed_Depth_KittiSplit_Full, self).__init__(
			args,
			images_root=root,
			flip_augmentations=flip_augmentations,
			preprocessing_crop=preprocessing_crop,
			crop_size=crop_size,
			num_examples=num_examples,
			index_file="index_txt/kitti_full.txt")