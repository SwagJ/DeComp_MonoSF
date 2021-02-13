from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging 

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms

def get_grid(x):
	grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
	grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
	grid = torch.cat([grid_H, grid_V], 1)
	grids_cuda = grid.float().requires_grad_(False).cuda()
	return grids_cuda


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

class WarpingLayer_Flow_Exp(nn.Module):
	def __init__(self):
		super(WarpingLayer_Flow_Exp, self).__init__()
 
	def forward(self, x, flow, disp, k1, input_size, exp):

		_, _, h_x, w_x = x.size()
		disp = interpolate2d_as(disp, x) * w_x
		exp = interpolate2d_as(exp, x)

		local_scale = torch.zeros_like(input_size)
		local_scale[:, 0] = h_x
		local_scale[:, 1] = w_x

		sceneflow = torch.cat([flow,disp*(exp-1)],dim=1)
		#print(flow.shape)

		pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
		_, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

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
		#	conv(32, 1, isReLU=False), 
		#	torch.nn.Sigmoid()
		#)
		self.conv_sf = conv(32, 2, isReLU=False)

	def forward(self, x):

		x_out = self.convs(x)
		sf = self.conv_sf(x_out)
		#disp1 = self.conv_d1(x_out) * 0.3
		
		return sf


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


class Feature_Decoder(nn.Module):
	def __init__(self,num_chs):
		super(Feature_Decoder, self).__init__()

		self.flow_decoder = conv(num_chs,num_chs,isReLU=False)
		self.disp_decoder = conv(num_chs,num_chs,isReLU=False)

	def forward(self,x):
		disp_feat = self.disp_decoder(x)
		flow_feat = self.flow_decoder(x)

		return disp_feat, flow_feat