from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging 
import numpy as np

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, flow2sf, flow2sf_dispC_v2
import collections

def get_grid(x):
	grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
	grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
	grid = torch.cat([grid_H, grid_V], 1)
	grids_cuda = grid.float().requires_grad_(False).cuda()
	return grids_cuda

def get_grid_exp(B,H,W):
	meshgrid_base = np.meshgrid(range(0,W), range(0,H))[::-1]
	basey = np.reshape(meshgrid_base[0],[1,1,1,H,W])
	basex = np.reshape(meshgrid_base[1],[1,1,1,H,W])
	grid = torch.tensor(np.concatenate((basex.reshape((-1,H,W,1)),basey.reshape((-1,H,W,1))),-1)).cuda().float()
	return grid.view(1,1,H,W,2)

def normalize_feature(feature_list):
	statistics = collections.defaultdict(list)
	for feature in feature_list:
		var, mean = torch.var_mean(feature, dim=[1,2,3], keepdim=True)
		statistics['mean'].append(mean)
		statistics['var'].append(var)

	statistics['mean'] = [torch.mean(torch.stack(statistics['mean']))] * 2
	statistics['var'] = [torch.mean(torch.stack(statistics['var']))] * 2
	statistics['std'] = [torch.sqrt(var + 1e-16) for var in statistics['var']]

	feature_list = [feature - mean for feature, mean in zip(feature_list, statistics['mean'])]
	feature_list = [feature / std for feature, std in zip(feature_list, statistics['std'])]

	return feature_list[0], feature_list[1]



class WarpingLayer_Flow(nn.Module):
	def __init__(self):
		super(WarpingLayer_Flow, self).__init__()

	def forward(self, x, flow):
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

class WarpingLayer_Flow_W_Mask(nn.Module):
	def __init__(self):
		super(WarpingLayer_Flow_W_Mask, self).__init__()

	def forward(self, x, flow):
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

		return x_warp * mask, mask


class WarpingLayer_SF(nn.Module):
	def __init__(self):
		super(WarpingLayer_SF, self).__init__()
 
	def forward(self, x, sceneflow, disp, k1, input_size):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask

class WarpingLayer_SF_DispC(nn.Module):
	def __init__(self):
		super(WarpingLayer_SF_DispC, self).__init__()
 
	def forward(self, x, flow, dispC, disp, k1, input_size):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		sceneflow = flow2sf_dispC_v2(flow, disp, dispC, k1, local_scale / input_size)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask

class WarpingLayer_SF_DispC_Flow(nn.Module):
	def __init__(self):
		super(WarpingLayer_SF_DispC_Flow, self).__init__()
		self._warping_layer = WarpingLayer_Flow()
 
	def forward(self, x, flow, dispC, disp, k1, input_size):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		disp_next = disp + dispC 
		#sceneflow = flow2sf_dispC_v2(flow, disp, dispC, k1, local_scale / input_size)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		pts1_next, k1_scale_next = pixel2pts_ms(k1, disp_next, local_scale / input_size)
		pts1_warpped = self._warping_layer(pts1_next, flow)
		sceneflow = pts1_warpped - pts1

		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask

class WarpingLayer_Flow_Exp(nn.Module):
	def __init__(self):
		super(WarpingLayer_Flow_Exp, self).__init__()
 
	def forward(self, x, flow, exp, disp, k1, input_size):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		sceneflow = flow2sf(flow, disp, exp, k1, local_scale / input_size)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask

class WarpingLayer_SF_PWC(nn.Module):
	def __init__(self):
		super(WarpingLayer_SF_PWC, self).__init__()
 
	def forward(self, x, sceneflow, disp, k1, input_size, ii):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		coef = 0.625 * (ii == 1) + 1.25 * (ii == 2) + 2.5 * (ii == 3) + 5 * (ii == 4)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow * coef, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask

class WarpingLayer_Flow_Exp_Plus(nn.Module):
	def __init__(self):
		super(WarpingLayer_Flow_Exp_Plus, self).__init__()
 
	def forward(self, x, flow, disp, k1, input_size, exp):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x
		exp = interpolate2d_as(exp, x) * w_x

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		sceneflow = torch.cat([flow,exp],dim=1)
		#print(flow.shape)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

		grid = coord1.transpose(1, 2).transpose(2, 3)
		x_warp = tf.grid_sample(x, grid)

		mask = torch.ones_like(x, requires_grad=False)
		mask = tf.grid_sample(mask, grid)
		mask = (mask >= 1.0).float()

		return x_warp * mask


def initialize_msra(modules):
	logging.info("Initializing MSRA")
	for layer in modules:
		if isinstance(layer, nn.Conv2d):
			nn.init.kaiming_normal_(layer.weight)
			if layer.bias is not None:
				nn.init.constant_(layer.bias, 0)

		elif isinstance(layer, nn.ConvTranspose2d):
			nn.init.kaiming_normal_(layer.weight)
			if layer.bias is not None:
				nn.init.constant_(layer.bias, 0)

		elif isinstance(layer, nn.LeakyReLU):
			pass

		elif isinstance(layer, nn.Sequential):
			pass


def upsample_outputs_as(input_list, ref_list):
	output_list = []
	for ii in range(0, len(input_list)):
		output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

	return output_list


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
	if isReLU:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
					  padding=((kernel_size - 1) * dilation) // 2, bias=True),
			nn.LeakyReLU(0.1, inplace=True)
		)
	else:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
					  padding=((kernel_size - 1) * dilation) // 2, bias=True)
		)

def convbn(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
	if isReLU:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
					  padding=((kernel_size - 1) * dilation) // 2, bias=True),
			nn.BatchNorm2d(out_planes),
			nn.LeakyReLU(0.1, inplace=True)
		)
	else:
		return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
					  padding=((kernel_size - 1) * dilation) // 2, bias=True),
			nn.BatchNorm2d(out_planes)
		)

class conv2DBatchNorm(nn.Module):
	def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
		super(conv2DBatchNorm, self).__init__()
		bias = not with_bn

		if dilation > 1:
			conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
								 padding=padding, stride=stride, bias=bias, dilation=dilation)

		else:
			conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
								 padding=padding, stride=stride, bias=bias, dilation=1)


		if with_bn:
			self.cb_unit = nn.Sequential(conv_mod,
										 nn.BatchNorm2d(int(n_filters)),)
		else:
			self.cb_unit = nn.Sequential(conv_mod,)

	def forward(self, inputs):
		outputs = self.cb_unit(inputs)
		return outputs

class conv2DBatchNormRelu(nn.Module):
	def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
		super(conv2DBatchNormRelu, self).__init__()
		bias = not with_bn
		if dilation > 1:
			conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
								 padding=padding, stride=stride, bias=bias, dilation=dilation)

		else:
			conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
								 padding=padding, stride=stride, bias=bias, dilation=1)

		if with_bn:
			self.cbr_unit = nn.Sequential(conv_mod,
										  nn.BatchNorm2d(int(n_filters)),
										  nn.LeakyReLU(0.1, inplace=True),)
		else:
			self.cbr_unit = nn.Sequential(conv_mod,
										  nn.LeakyReLU(0.1, inplace=True),)

	def forward(self, inputs):
		outputs = self.cbr_unit(inputs)
		return outputs

class pyramidPooling(nn.Module):

	def __init__(self, in_channels, with_bn=True, levels=4):
		super(pyramidPooling, self).__init__()
		self.levels = levels

		self.paths = []
		for i in range(levels):
			self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
		self.path_module_list = nn.ModuleList(self.paths)
		self.relu = nn.LeakyReLU(0.1, inplace=True)
	
	def forward(self, x):
		h, w = x.shape[2:]

		k_sizes = []
		strides = []
		for pool_size in np.linspace(1,min(h,w)//2,self.levels,dtype=int):
			k_sizes.append((int(h/pool_size), int(w/pool_size)))
			strides.append((int(h/pool_size), int(w/pool_size)))
		k_sizes = k_sizes[::-1]
		strides = strides[::-1]

		pp_sum = x

		for i, module in enumerate(self.path_module_list):
			out = tf.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
			out = module(out)
			out = tf.interpolate(out, size=(h,w), mode='bilinear')
			pp_sum = pp_sum + 1./self.levels*out
		pp_sum = self.relu(pp_sum/2.)

		return pp_sum

class ConvBlock(nn.Module):
	"""Layer to perform a convolution followed by ELU
	"""
	def __init__(self, in_channels, out_channels):
		super(ConvBlock, self).__init__()

		self.conv = Conv3x3(in_channels, out_channels)
		self.nonlin = nn.ELU(inplace=True)

	def forward(self, x):
		out = self.conv(x)
		out = self.nonlin(out)
		return out


class upconv(nn.Module):
	def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
		super(upconv, self).__init__()
		self.scale = scale
		self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

	def forward(self, x):
		x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
		return self.conv1(x)


class FeatureExtractor(nn.Module):
	def __init__(self, num_chs):
		super(FeatureExtractor, self).__init__()
		self.num_chs = num_chs
		self.convs = nn.ModuleList()

		for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
			layer = nn.Sequential(
				conv(ch_in, ch_out, stride=2),
				conv(ch_out, ch_out)
			)
			self.convs.append(layer)

	def forward(self, x):
		feature_pyramid = []
		for conv in self.convs:
			x = conv(x)
			feature_pyramid.append(x)

		return feature_pyramid[::-1]


class MonoSceneFlowDecoder(nn.Module):
	def __init__(self, ch_in):
		super(MonoSceneFlowDecoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_sf = conv(32, 3, isReLU=False)
		self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out)

		return x_out, sf, disp1

class MonoSF_DispC_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(MonoSF_DispC_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_flow = conv(32, 2, isReLU=False)
		self.conv_dispC = conv(32, 1, isReLU=False)
		self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		sf = self.conv_flow(x_out)
		disp1 = self.conv_d1(x_out)
		dispC = self.conv_dispC(x_out) * 0.1

		return x_out, sf, disp1, dispC

class MonoFlow_Disp_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(MonoFlow_Disp_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_sf = conv(32, 2, isReLU=False)
		self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out)

		return x_out, sf, disp1

class MonoFlow_DispC_Decoder_v1_1(nn.Module):
	def __init__(self, ch_in):
		super(MonoFlow_DispC_Decoder_v1_1, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_sf = conv(32, 2, isReLU=False)
		self.conv_dc = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		flo = self.conv_sf(x_out)
		dispC = self.conv_dc(x_out) * 0.3

		return x_out, flo, dispC

class MonoFlow_DispC_Decoder_v1_2(nn.Module):
	def __init__(self, ch_in):
		super(MonoFlow_DispC_Decoder_v1_2, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_sf = conv(32, 2, isReLU=False)
		self.conv_dc = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		flo = self.conv_sf(x_out)
		dispC = self.conv_dc(x_out)

		return x_out, flo, dispC

class MonoSF_Disp_Exp_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(MonoSF_Disp_Exp_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_flow = conv(32, 2, isReLU=False)
		self.conv_d1 = conv(32, 1, isReLU=False)
		self.conv_exp1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		flow = self.conv_flow(x_out)
		disp1 = self.conv_d1(x_out)
		exp1 = self.conv_exp1(x_out)

		return x_out, flow, disp1, exp1


class ContextNetwork(nn.Module):
	def __init__(self, ch_in, is_exp=False):
		super(ContextNetwork, self).__init__()
		self.is_exp = is_exp

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		if is_exp == True:
			self.conv_sf = conv(32, 2, isReLU=False)
			self.conv_exp1 = nn.Sequential(
				conv(32, 1, isReLU=False), 
				torch.nn.Sigmoid()
			)
		else:
			self.conv_sf = conv(32, 3, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		if self.is_exp == True:
			exp1 = self.conv_exp1(x_out) * 0.1
			return sf, disp1, exp1
		else:
			return sf, disp1

class ContextNetwork_DispC_SF(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_DispC_SF, self).__init__()
		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		self.conv_sf = conv(32, 2, isReLU=False)
		self.conv_dispC = conv(32, 1, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		dispC = self.conv_dispC(x_out) * 0.1
		
		return sf, disp1, dispC

class ContextNetwork_Flow_Disp(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Flow_Disp, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		self.conv_sf = conv(32, 2, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		
		return sf, disp1

class ContextNetwork_Flow_DispC_v1_1(nn.Module):
	def __init__(self, ch_in, is_exp=False):
		super(ContextNetwork_Flow_DispC_v1_1, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_dc = conv(32, 1, isReLU=False)
		self.conv_sf = conv(32, 2, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		flo = self.conv_sf(x_out)
		dispC = self.conv_dc(x_out) * 0.3
		
		return flo, dispC

class ContextNetwork_Flow_DispC_v1_2(nn.Module):
	def __init__(self, ch_in, is_exp=False):
		super(ContextNetwork_Flow_DispC_v1_2, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_dc = conv(32, 1, isReLU=False)
		self.conv_sf = conv(32, 2, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		flo = self.conv_sf(x_out)
		dispC = self.conv_dc(x_out)
		
		return flo, dispC


class Conv3x3(nn.Module):
	"""Layer to pad and convolve input
	"""
	def __init__(self, in_channels, out_channels, use_refl=True):
		super(Conv3x3, self).__init__()

		if use_refl:
			self.pad = nn.ReflectionPad2d(1)
		else:
			self.pad = nn.ZeroPad2d(1)
		self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

	def forward(self, x):
		out = self.pad(x)
		out = self.conv(out)
		return out

class Disp_Decoder_Skip_Connection(nn.Module):
	def __init__(self,ch_in):
		super(Disp_Decoder_Skip_Connection, self).__init__()

		self.upconvs = nn.ModuleList()
		self.ch_in = ch_in[::-1]
		self.ch_dec = [16, 32, 64, 128, 192, 256, 256]
		#print("ch_in:", self.ch_in)
		self.sigmoid = nn.Sigmoid()
		self.disp_decoders = nn.ModuleList()
		for i in range(6):
			#print("ii:",i)
			upconvs_now = nn.ModuleList()
			#print("conv1_block dim:",self.ch_in[5-i], self.ch_dec[5-i])
			upconvs_now.append(ConvBlock(self.ch_dec[6-i],self.ch_dec[5-i]))
			if i != 5:
				upconvs_now.append(ConvBlock(self.ch_in[i+1] + self.ch_dec[5-i], self.ch_dec[5-i]))
				#print("conv2_block dim:",self.ch_in[i+1] + self.ch_dec[5-i], self.ch_dec[5-i])
			else:
				upconvs_now.append(ConvBlock(self.ch_dec[5-i], self.ch_dec[5-i]))
				#print("conv2_block dim:",self.ch_dec[5-i], self.ch_dec[5-i])

			self.upconvs.append(upconvs_now)
			self.disp_decoders.append(Conv3x3(self.ch_dec[5-i],1))

	def forward(self, input_features):
		disps = []

		x = input_features[0]
		#print("input feature shape:", input_features[0].shape,input_features[1].shape, input_features[2].shape, input_features[3].shape, input_features[4].shape,input_features[5].shape, input_features[6].shape)
		for i in range(6):
			#print("ii:", i)
			scale = 5 - i
			x = self.upconvs[i][0](x)
			x = [tf.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)]
			if i != 5:
				x += [input_features[i+1]]
			x = torch.cat(x,1)
			x = self.upconvs[i][1](x)

			disps.append(self.sigmoid(self.disp_decoders[i](x)))

		return disps



class Flow_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(Flow_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_flow = conv(32, 2, isReLU=False)
		#self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		flow = self.conv_flow(x_out)

		return x_out, flow

class DispC_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(DispC_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 96),
			conv(96, 64),
			conv(64, 32)
		)
		self.conv_flow = conv(32, 1, isReLU=False)
		#self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		flow = self.conv_flow(x_out)

		return x_out, flow

class Disp_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(Disp_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 64),
			conv(64, 32)
		)
		self.conv_disp = conv(32, 1, isReLU=False)
		#self.conv_d1 = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		disp = self.conv_disp(x_out)

		return x_out, disp

class Disp_DispC_Decoder(nn.Module):
	def __init__(self, ch_in):
		super(Disp_DispC_Decoder, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 64),
			conv(64, 32)
		)
		self.conv_disp = conv(32, 1, isReLU=False)
		self.conv_dc = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		disp = self.conv_disp(x_out)
		dispC = self.conv_dc(x_out)

		return x_out, disp, dispC

class Disp_DispC_Decoder_v2_2(nn.Module):
	def __init__(self, ch_in):
		super(Disp_DispC_Decoder_v2_2, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128),
			conv(128, 128),
			conv(128, 64),
			conv(64, 32)
		)
		self.conv_disp = conv(32, 1, isReLU=False)
		self.conv_dc = conv(32, 1, isReLU=False)

	def forward(self, x):
		x_out = self.convs(x)
		disp = self.conv_disp(x_out)
		dispC = self.conv_dc(x_out) * 0.1

		return x_out, disp, dispC

class ContextNetwork_Flow(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Flow, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		#self.conv_d1 = nn.Sequential(
		#   conv(32, 1, isReLU=False), 
		#   torch.nn.Sigmoid()
		#)
		self.conv_sf = conv(32, 2, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		#disp1 = self.conv_d1(x_out) * 0.3
		
		return sf

class ContextNetwork_DispC(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_DispC, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		#self.conv_d1 = nn.Sequential(
		#   conv(32, 1, isReLU=False), 
		#   torch.nn.Sigmoid()
		#)
		self.conv_sf = conv(32, 1, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		#disp1 = self.conv_d1(x_out) * 0.3
		
		return sf

class ContextNetwork_Exp(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Exp, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		#self.conv_d1 = nn.Sequential(
		#   conv(32, 1, isReLU=False), 
		#   torch.nn.Sigmoid()
		#)
		self.conv_sf = conv(32, 1, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		exp = self.conv_sf(x_out)
		#disp1 = self.conv_d1(x_out) * 0.3
		
		return exp


class ContextNetwork_Disp(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Disp, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		#self.conv_sf = conv(32, 1, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		#sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		
		return disp1

class ContextNetwork_Disp_DispC(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Disp_DispC, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		self.conv_dc = conv(32, 1, isReLU=True)

	def forward(self, x):

		x_out = self.convs(x)
		#sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		dispC = self.conv_dc(x_out)
		
		return disp1, dispC

class ContextNetwork_Disp_DispC_v2_2(nn.Module):
	def __init__(self, ch_in):
		super(ContextNetwork_Disp_DispC_v2_2, self).__init__()

		self.convs = nn.Sequential(
			conv(ch_in, 128, 3, 1, 1),
			conv(128, 128, 3, 1, 2),
			conv(128, 128, 3, 1, 4),
			conv(128, 96, 3, 1, 8),
			conv(96, 64, 3, 1, 16),
			conv(64, 32, 3, 1, 1)
		)
		self.conv_d1 = nn.Sequential(
			conv(32, 1, isReLU=False), 
			torch.nn.Sigmoid()
		)
		self.conv_dc = conv(32, 1, isReLU=True)

	def forward(self, x):

		x_out = self.convs(x)
		#sf = self.conv_sf(x_out)
		disp1 = self.conv_d1(x_out) * 0.3
		dispC = self.conv_dc(x_out) * 0.1
		
		return disp1, dispC


class Feature_Decoder(nn.Module):
	def __init__(self,num_chs):
		super(Feature_Decoder, self).__init__()

		self.flow_decoder = conv(num_chs,num_chs,isReLU=False)
		self.disp_decoder = conv(num_chs,num_chs,isReLU=False)

	def forward(self,x):
		disp_feat = self.disp_decoder(x)
		flow_feat = self.flow_decoder(x)

		return disp_feat, flow_feat

class Feature_Decoder_ppV1(nn.Module):
	def __init__(self,num_chs):
		super(Feature_Decoder_ppV1, self).__init__()

		self.flow_decoder = conv(num_chs,num_chs,isReLU=False)
		self.disp_decoder = conv(num_chs,num_chs,isReLU=False)
		self.flow_x_decoder = conv(num_chs,num_chs,isReLU=False)

	def forward(self,x):
		disp_feat = self.disp_decoder(x)
		flow_feat = self.flow_decoder(x)
		exp_feat = self.flow_x_decoder(x)

		return disp_feat, flow_feat, exp_feat

###################################################
#
# Bottom: pyramid pooling + a predictor
# Upper: skip connector + a predictor
#
###################################################
class Flow_x_Decoder_ppV1_Skip(nn.Module):
	def __init__(self):
		super(Flow_x_Decoder_ppV1_Skip, self).__init__()
		self.pyramid_pooling = pyramidPooling(256, levels=3)
		#print("ch_in:", self.ch_in)
		self.upconv6 = conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=192,
												 padding=1, stride=1)
		self.upconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
												 padding=1, stride=1)
		self.upconv4 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
												 padding=1, stride=1)
		self.iconv5 = conv2DBatchNormRelu(in_channels=384, k_size=3, n_filters=128,
												 padding=1, stride=1)
		self.iconv4 = conv2DBatchNormRelu(in_channels=256, k_size=3, n_filters=96,
												 padding=1, stride=1)
		self.iconv3 = conv2DBatchNormRelu(in_channels=96 + 64, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.iconv2 = nn.Sequential(conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
												 padding=1, stride=1),
									nn.Conv2d(64, 1,kernel_size=3, stride=1, padding=1, bias=True))

		self.proj6 = nn.Conv2d(256, 1,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj5 = nn.Conv2d(128, 1,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj4 = nn.Conv2d(96, 1,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj3 = nn.Conv2d(64, 1,kernel_size=3, stride=1, padding=1, bias=True)

	def forward(self, input_features):
		x6, x5, x4, x3, x2, x1, x0 = input_features
		conv6 = self.pyramid_pooling(x6)
		pred6 = self.proj6(conv6)

		conv6u = tf.interpolate(conv6,[x5.size()[2],x5.size()[3]], mode='bilinear')
		conv5 = self.iconv5(torch.cat((x5,self.upconv6(upconv6u)),dim=1))
		pred5 = self.proj5(conv5)

		conv5u = tf.interpolate(conv5,[x4.size()[2],x4.size()[3]], mode='bilinear')
		conv4 = self.iconv4(torch.cat((x4,self.upconv5(upconv5u)),dim=1))
		pred4 = self.proj4(conv4)

		conv4u = tf.interpolate(conv4,[x4.size()[2],x4.size()[3]], mode='bilinear')
		conv3 = self.iconv3(torch.cat((x4,self.upconv4(upconv4u)),dim=1))
		pred3 = self.proj3(conv3)

		conv3u = tf.interpolate(conv3,[x3.size()[2],x3.size()[3]], mode='bilinear')
		pred2 = self.iconv2(torch.cat((x3,self.upconv6(upconv3u)),dim=1))
		#pred2 = self.proj2(conv2)

		preds = [pred6, pred5, pred4, pred3, pred2]

		return preds


class Exp_Decoder_ppV1_Dense(nn.Module):
	def __init__(self, ch_in, is_bottom):
		super(Exp_Decoder_ppV1_Dense, self).__init__()
		self.is_bottom = is_bottom
		if is_bottom:
			self.pooling = pyramidPooling(ch_in, levels=3, with_bn=False)
			self.convs = conv(ch_in, 32)
		else:
			self.convs = nn.Sequential(
				conv(ch_in, 128),
				conv(128, 128),
				conv(128, 96),
				conv(96, 64),
				conv(64, 32)
			)
		self.conv_exp = conv(32, 1, isReLU=False)

	def forward(self, x):
		if self.is_bottom:
			x = self.pooling(x)
		x_out = self.convs(x)
		exp = self.conv_exp(x_out)

		return x_out, exp


def affine(flow, pw=1):
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
		return exp, Error, mask