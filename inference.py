from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImages import saveTensorToImages
import sys
import importlib
import cv2
import skimage.io as io
import png
import logging
import torch.nn.functional as tf

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
	format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")

def read_image_as_byte(filename):
	return io.imread(filename)

def read_png_disp(disp_file):
	disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
	disp_np = np.expand_dims(disp_np, axis=2)
	mask_disp = (disp_np > 0).astype(np.float64)
	return disp_np, mask_disp


def interpolate2d_as(inputs, target_as, mode="bilinear"):
	_, _, h, w = target_as.size()
	return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

class KITTI_Raw_Depth(Dataset):
	def __init__(self,
				 images_root=None,
				 index_file=None):

		self.images_root = images_root

		path_dir = os.path.dirname(os.path.realpath(__file__))
		path_index_file = os.path.join(path_dir, 'kitti_full.txt')

		if not os.path.exists(path_index_file):
			raise ValueError("Index File '%s' not found!", path_index_file)
		index_file = open(path_index_file, 'r')

		## loading image -----------------------------------
		if not os.path.isdir(images_root):
			raise ValueError("Image directory '%s' not found!")

		filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
		self._image_list = []
		self._depth_list = []
		self._outdir_l = []
		self._outdir_r = []
		view1 = 'image_02/data'
		view2 = 'image_03/data'
		depth1 = 'proj_depth/groundtruth/image_02'
		depth2 = 'proj_depth/groundtruth/image_03'
		ext = '.png'
		
		for item in filename_list:
			date = item[0][:10]
			scene = item[0]
			idx_src = item[1]
			name_l = os.path.join(images_root, date, scene, view1, idx_src) + ext
			name_r = os.path.join(images_root, date, scene, view2, idx_src) + ext
			depth_l = os.path.join(images_root, date, scene, depth1, idx_src) + ext
			depth_r = os.path.join(images_root, date, scene, depth2, idx_src) + ext
			out_file_l = os.path.join(images_root, date, scene, 'completed_depth', 'image_02', idx_src) + ext
			out_file_r = os.path.join(images_root, date, scene, 'completed_depth', 'image_03', idx_src) + ext

			if os.path.isfile(name_l) and os.path.isfile(name_r) and os.path.isfile(depth_l) and os.path.isfile(depth_r):
				self._image_list.append([name_l, name_r])
				self._depth_list.append([depth_l, depth_r])
				self._outdir_l.append(out_file_l)
				self._outdir_r.append(out_file_r)

			outdir_l = os.path.join(images_root, date, scene, 'completed_depth', 'image_02')
			outdir_r = os.path.join(images_root, date, scene, 'completed_depth', 'image_03')

			if not os.path.exists(outdir_l):
				os.makedirs(outdir_l)
				logging.info("Creating directory for left images:" + outdir_l)
			if not os.path.exists(outdir_r):
				os.makedirs(outdir_r)
				logging.info("Creating directory for left images:" + outdir_r)

		if len(self._image_list) != len(self._depth_list):
			raise ValueError("Image and Depth Incosistency!!! Double Check")

		for idx, _ in enumerate(self._image_list):
			#print(self._image_list[idx])
			for sub_idx, _ in enumerate(self._image_list[idx]):
				if not os.path.isfile(self._image_list[idx][sub_idx]):
					raise ValueError("Image File not exist: %s", self._image_list[idx])
				if not os.path.isfile(self._depth_list[idx][sub_idx]):
					raise ValueError("Depth File not exist: %s", self._depth_list[idx])

		self._size = len(self._image_list)

		## loading calibration matrix

	def __getitem__(self, index):
		index = index % self._size

		# read images and flow
		# im_l1, im_l2, im_r1, im_r2

		# example filename
		im_l_filename = self._image_list[index][0]
		im_r_filename = self._image_list[index][1]
		depth_l_filename = self._depth_list[index][0]
		depth_r_filename = self._depth_list[index][1]
		out_file_l = self._outdir_l[index]
		out_file_r = self._outdir_r[index]

		bname_im_l = os.path.basename(im_l_filename)
		bname_im_r = os.path.basename(im_r_filename)
		bname_depth_l = os.path.basename(depth_l_filename)
		bname_depth_r = os.path.basename(depth_r_filename)
		bname_out_l = os.path.basename(out_file_l)
		bname_out_r = os.path.basename(out_file_r)

		if not((bname_im_r == bname_out_r) and (bname_out_r==bname_depth_r) and (bname_depth_r==bname_im_r)):
			print(bname_im_r,bname_out_r,bname_depth_r)
			raise ValueError("right image pair Incosistency!")

		if not((bname_im_l == bname_out_l) and (bname_out_l==bname_depth_l) and (bname_depth_l==bname_im_l)):
			raise ValueError("left image pair Incosistency!")

		# read image
		rgb_l = Image.open(im_l_filename)
		rgb_r = Image.open(im_r_filename)
		depth_l = Image.open(depth_l_filename)
		depth_r = Image.open(depth_r_filename)

		rgb_l = np.array(rgb_l)
		rgb_r = np.array(rgb_r)
		depth_l = np.array(depth_l)
		depth_r = np.array(depth_r)
		# get confidence map
		C_l = (depth_l > 0).astype(float)
		C_r = (depth_r > 0).astype(float)
		# preprocess
		depth_l = depth_l / 256
		depth_r = depth_r / 256
		rgb_l = rgb_l / 256
		rgb_r = rgb_r / 256

		C_l = np.expand_dims(C_l, 0) 
		depth_l = np.expand_dims(depth_l, 0) 
		rgb_l = np.transpose(rgb_l,(2,0,1))
		C_r = np.expand_dims(C_r, 0) 
		depth_r = np.expand_dims(depth_r, 0) 
		rgb_r = np.transpose(rgb_r,(2,0,1))

		C_l = torch.tensor(C_l, dtype=torch.float)
		depth_l = torch.tensor(depth_l, dtype=torch.float)
		rgb_l = torch.tensor(rgb_l, dtype=torch.float)
		C_r = torch.tensor(C_r, dtype=torch.float)
		depth_r = torch.tensor(depth_r, dtype=torch.float)
		rgb_r = torch.tensor(rgb_r, dtype=torch.float)
		
		input_dict = {
			"out_file_l": out_file_l,
			"out_file_r": out_file_r,
			"C_l": C_l,
			"depth_l": depth_l,
			"rgb_l": rgb_l,
			"C_r": C_r,
			"depth_r": depth_r,
			"rgb_r": rgb_r
		}

		# random flip

		return input_dict

	def __len__(self):
		return self._size


def main():
	training_ws_path = 'workspace/'
	exp_dir = os.path.join(training_ws_path, 'exp_guided_nconv_cnn_l1')
	images_root = '/disk_hdd/kitti_full'

	# Add the experiment's folder to python path
	sys.path.append(exp_dir)
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
	logging.info("Using device: " + str(device))
	
	#Add checkpoint path
	chkpt_path = os.path.join(exp_dir, 'checkpoints/CNN_ep0040.pth.tar')
	logging.info("Checkpoint exists: " + str(os.path.exists(chkpt_path)))

	# init model
	f = importlib.import_module('network_'+'exp_guided_nconv_cnn_l1')
	model = f.CNN(pos_fn = "None").to(device)

	checkpoint_dict = torch.load(chkpt_path)

	model.load_state_dict(checkpoint_dict['net'])
	logging.info("Weight loaded.")	
	model.eval()

	kitti_dataset = KITTI_Raw_Depth(images_root=images_root)
	dataloader = DataLoader(kitti_dataset,batch_size=1,shuffle=False, num_workers=4)
	logging.info("Model in Inference.")
	num_images = len(kitti_dataset)
	logging.info("Total number of images to process: {}".format(num_images))
	with torch.no_grad():
		for (i,input_dict) in enumerate(dataloader):
			C_l = input_dict['C_l'].to(device)
			depth_l = input_dict['depth_l'].to(device)
			rgb_l = input_dict['rgb_l'].to(device)
			C_r = input_dict['C_r'].to(device)
			depth_r = input_dict['depth_r'].to(device)
			rgb_r = input_dict['rgb_r'].to(device)
			out_file_l = input_dict['out_file_l']
			out_file_r = input_dict['out_file_r']
		
			outputs_l,_ = model(depth_l, C_l, rgb_l)
			outputs_r,_ = model(depth_r, C_r, rgb_r)

			outputs_l = interpolate2d_as(outputs_l, rgb_l, mode="bilinear")
			outputs_r = interpolate2d_as(outputs_r, rgb_r, mode="bilinear")

			outputs_l = outputs_l.data
			outputs_l *=256
			outputs_r = outputs_r.data
			outputs_r *=256

			d_l = outputs_l[0,:,:,:].detach().data.cpu().numpy() 
			d_l = np.transpose(d_l, (1,2,0)).astype(np.uint16)
			d_r = outputs_r[0,:,:,:].detach().data.cpu().numpy() 
			d_r = np.transpose(d_r, (1,2,0)).astype(np.uint16)

			cv2.imwrite(out_file_l[0],d_l, [cv2.IMWRITE_PNG_COMPRESSION, 4])
			cv2.imwrite(out_file_r[0],d_r, [cv2.IMWRITE_PNG_COMPRESSION, 4])
			logging.info("Writing Image pair:" + out_file_r[0])

			if (i % 100 == 0):
				logging.info("Processed {} images out of Total {} images".format(i,num_images))


		logging.info("Process finished!")



if __name__ == "__main__":
	main()
