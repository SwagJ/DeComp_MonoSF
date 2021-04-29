from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb

from .modules_sceneflow import ConvBlock, Conv3x3
from .model_monosceneflow import MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res, MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm

class MonoFlowDispKernel(nn.Module):
	def __init__(self, args):
		super(MonoFlowDispKernel, self).__init__()
		self._args = args
		self._model = MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res(args)

	def forward(self, input_dict):
		return self._model(input_dict)

class MonoFlowDispKernel_v2(nn.Module):
	def __init__(self, args):
		super(MonoFlowDispKernel_v2, self).__init__()
		self._args = args
		self._model = MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm(args)

	def forward(self, input_dict):
		return self._model(input_dict)

class residualBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1,with_bn=True):
		super(residualBlock, self).__init__()
		if dilation > 1:
			padding = dilation
		else:
			padding = 1

		if with_bn:
			self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation)
			self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
		else:
			self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation,with_bn=False)
			self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
		self.downsample = downsample
		self.relu = nn.LeakyReLU(0.1, inplace=True)

	def forward(self, x):
		residual = x

		out = self.convbnrelu1(x)
		out = self.convbn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		return self.relu(out)

def get_grid(B,H,W):
	meshgrid_base = np.meshgrid(range(0,W), range(0,H))[::-1]
	basey = np.reshape(meshgrid_base[0],[1,1,1,H,W])
	basex = np.reshape(meshgrid_base[1],[1,1,1,H,W])
	grid = torch.tensor(np.concatenate((basex.reshape((-1,H,W,1)),basey.reshape((-1,H,W,1))),-1)).cuda().float()
	return grid.view(1,1,H,W,2)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
	return nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
						padding=padding, dilation=dilation, bias=True),
			nn.BatchNorm2d(out_planes),
			nn.LeakyReLU(0.1,inplace=True))

def conv_no_bn(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
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


class pyramidPooling(nn.Module):

	def __init__(self, in_channels, with_bn=True, levels=4):
		super(pyramidPooling, self).__init__()
		self.levels = levels

		self.paths = []
		for i in range(levels-1):
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
			out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
			#print(out.shape)
			out = module(out)
			out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
			pp_sum = pp_sum + 1./self.levels*out
		pp_sum = self.relu(pp_sum/2.)

		return pp_sum

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




class bfmodule(nn.Module):
	def __init__(self, inplanes, outplanes):
		super(bfmodule, self).__init__()
		self.proj = conv2DBatchNormRelu(in_channels=inplanes,k_size=1,n_filters=64,padding=0,stride=1)
		self.inplanes = 64
		# Vanilla Residual Blocks
		self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
		self.res_block5 = self._make_layer(residualBlock,64,1,stride=2)
		self.res_block6 = self._make_layer(residualBlock,64,1,stride=2)
		self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
		self.pyramid_pooling = pyramidPooling(128, levels=3)
		# Iconvs
		self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.upconv5 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
												 padding=1, stride=1)
		self.upconv4 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
												 padding=1, stride=1)
		self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
												 padding=1, stride=1)
		self.iconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.iconv4 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.iconv3 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
												 padding=1, stride=1)
		self.iconv2 = nn.Sequential(conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
												 padding=1, stride=1),
									nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True))

		self.proj6 = nn.Conv2d(128, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj5 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj4 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
		self.proj3 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if hasattr(m.bias,'data'):
					m.bias.data.zero_()
	   

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
												 kernel_size=1, stride=stride, bias=False),
									   nn.BatchNorm2d(planes * block.expansion),)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		proj = self.proj(x) # 4x
		rconv3 = self.res_block3(proj) #8x
		conv4 = self.res_block5(rconv3) #16x
		conv5 = self.res_block6(conv4) #32x
		conv6 = self.res_block7(conv5) #64x
		conv6 = self.pyramid_pooling(conv6) #64x
		pred6 = self.proj6(conv6)

		conv6u = F.interpolate(conv6, [conv5.size()[2],conv5.size()[3]], mode='bilinear', align_corners=True)
		concat5 = torch.cat((conv5,self.upconv6(conv6u)),dim=1) 
		conv5 = self.iconv5(concat5) #32x
		pred5 = self.proj5(conv5)

		conv5u = F.interpolate(conv5, [conv4.size()[2],conv4.size()[3]], mode='bilinear', align_corners=True)
		concat4 = torch.cat((conv4,self.upconv5(conv5u)),dim=1)
		conv4 = self.iconv4(concat4) #16x
		pred4 = self.proj4(conv4)

		conv4u = F.interpolate(conv4, [rconv3.size()[2],rconv3.size()[3]], mode='bilinear', align_corners=True)
		concat3 = torch.cat((rconv3,self.upconv4(conv4u)),dim=1)
		conv3 = self.iconv3(concat3) # 8x
		pred3 = self.proj3(conv3)

		conv3u = F.interpolate(conv3, [x.size()[2],x.size()[3]], mode='bilinear', align_corners=True)
		concat2 = torch.cat((proj,self.upconv3(conv3u)),dim=1)
		pred2 = self.iconv2(concat2)  # 4x

		return pred2, pred3, pred4, pred5, pred6


class Expansion_Decoder(nn.Module):
	def __init__(self, args, exp_unc=False):

		super(Expansion_Decoder, self).__init__()

		#self.f3d2v1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2v2 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2v3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v5 = conv(64,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v6 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2 = bfmodule(128-64,1)

		# depth change net
		self.dcnetv1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.dcnetv2 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.dcnetv3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.dcnetv4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.dcnetv5 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.dcnetv6 = conv(4,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		if exp_unc:       
			self.dcnet = bfmodule(128,2)
		else:
			self.dcnet = bfmodule(128,1)

	def affine(self,pref,flow, pw=1):
		b,_,lh,lw=flow.shape
		ptar = pref + flow
		pw = 1
		pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
		ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
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

	def affine_mask(self,pref,flow, pw=3):
		"""
		pref: reference coordinates
		pw: patch width
		"""
		flmask = flow[:,2:]
		flow = flow[:,:2]
		b,_,lh,lw=flow.shape
		ptar = pref + flow
		pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
		ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
		
		conf_flow = flmask
		conf_flow = F.unfold(conf_flow,(pw*2+1,pw*2+1), padding=(pw)).view(b,1,(pw*2+1)**2,lh,lw)
		count = conf_flow.sum(2,keepdims=True)
		conf_flow = ((pw*2+1)**2)*conf_flow / count
		pref = pref * conf_flow
		ptar = ptar * conf_flow
	
		pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
		ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

		prefprefT = pref.matmul(pref.permute(0,2,1))
		ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
		ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

		Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
		Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

		Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
		exp = Avol.sqrt()
		mask = (exp>0.5) & (exp<2) & (Error<0.2) & (flmask.bool()) & (count[:,0]>4)
		mask = mask[:,0]

		exp = exp.clamp(0.5,2)
		exp[Error>0.2]=1
		return exp, Error, mask


	def forward(self, flow2, c12, im0):
		# flow 2 is top level flow:
		# c12 is top level feature 
		b,_,h,w = flow2.shape 
		exp2,err2,_ = self.affine(get_grid(b,h,w)[:,0].permute(0,3,1,2).repeat(b,1,1,1).clone(), flow2.detach(),pw=1)
		x = torch.cat((
					   self.f3d2v2(-exp2.log()),
					   self.f3d2v3(err2),
					   ),1)
		dchange2 = -exp2.log()+1./200*self.f3d2(x)[0]

		# depth change net
		#print(c12.shape)
		#print(flow2.shape)
		iexp2 = F.interpolate(dchange2.clone(), [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)
		x = torch.cat((self.dcnetv1(c12.detach()),
					   self.dcnetv2(dchange2.detach()),
					   self.dcnetv3(-exp2.log()),
					   self.dcnetv4(err2),
					),1)
		dcneto = 1./200*self.dcnet(x)[0]
		dchange2 = dchange2.detach() + dcneto[:,:1]

		flow2 = F.interpolate(flow2.detach(), [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)
		dchange2 = F.interpolate(dchange2, [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)

		return dchange2,iexp2, flow2

		# if self.training:
		# 	flowl0 = disc_aux[0].permute(0,3,1,2).clone()
		# 	gt_depth = disc_aux[2][:,:,:,0]
		# 	gt_f3d =  disc_aux[2][:,:,:,4:7].permute(0,3,1,2).clone()
		# 	gt_dchange = (1+gt_f3d[:,2]/gt_depth)
		# 	maskdc = (gt_dchange < 2) & (gt_dchange > 0.5) & disc_aux[1]

		# 	gt_expi,gt_expi_err,maskoe = self.affine_mask(get_grid(b,4*h,4*w)[:,0].permute(0,3,1,2).repeat(b,1,1,1), flowl0,pw=3)
		# 	gt_exp = 1./gt_expi[:,0]

		# 	loss =  0.1* (dchange2[:,0]-gt_dchange.log()).abs()[maskdc].mean()
		# 	loss += 0.1* (iexp2[:,0]-gt_exp.log()).abs()[maskoe].mean()
		# 	return flow2*4, flow3*8,flow4*16,flow5*32,flow6*64,loss, dchange2[:,0], iexp2[:,0]

		# else:
		# 	return flow2,  dchange2[0,0], iexp2[0,0]


class DispC_Decoder(nn.Module):
	def __init__(self, args, exp_unc=False):

		super(DispC_Decoder, self).__init__()

		#self.f3d2v1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2v2 = conv(64,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2v3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v5 = conv(64,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.f3d2v6 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		self.f3d2 = bfmodule(128-64,1)

		# depth change net
		# self.dcnetv1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		# self.dcnetv2 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		# self.dcnetv3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		# self.dcnetv4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.dcnetv5 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		#self.dcnetv6 = conv(4,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
		if exp_unc:       
			self.dcnet = bfmodule(128,2)
		else:
			self.dcnet = bfmodule(128,1)

	def forward(self, init_dispC, c12, im0):
		# flow 2 is top level flow:
		# c12 is top level feature 
		b,_,h,w = init_dispC.shape 
		x = torch.cat((
					   self.f3d2v2(c12),
					   self.f3d2v3(init_dispC),
					   ),1)
		dchange2 = self.f3d2(x)[0]

		# depth change net
		#print(c12.shape)
		#print(flow2.shape)
		# iexp2 = F.interpolate(dchange2.clone(), [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)
		# x = torch.cat((self.dcnetv1(c12.detach()),
		# 			   self.dcnetv2(dchange2.detach()),
		# 			   self.dcnetv3(-exp2.log()),
		# 			   self.dcnetv4(err2),
		# 			),1)
		# dcneto = 1./200*self.dcnet(x)[0]
		# dchange2 = dchange2.detach() + dcneto[:,:1]

		# flow2 = F.interpolate(flow2.detach(), [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)
		dispC = F.interpolate(dchange2, [im0.size()[2],im0.size()[3]], mode='bilinear', align_corners=True)

		return dispC

		# if self.training:
		# 	flowl0 = disc_aux[0].permute(0,3,1,2).clone()
		# 	gt_depth = disc_aux[2][:,:,:,0]
		# 	gt_f3d =  disc_aux[2][:,:,:,4:7].permute(0,3,1,2).clone()
		# 	gt_dchange = (1+gt_f3d[:,2]/gt_depth)
		# 	maskdc = (gt_dchange < 2) & (gt_dchange > 0.5) & disc_aux[1]

		# 	gt_expi,gt_expi_err,maskoe = self.affine_mask(get_grid(b,4*h,4*w)[:,0].permute(0,3,1,2).repeat(b,1,1,1), flowl0,pw=3)
		# 	gt_exp = 1./gt_expi[:,0]

		# 	loss =  0.1* (dchange2[:,0]-gt_dchange.log()).abs()[maskdc].mean()
		# 	loss += 0.1* (iexp2[:,0]-gt_exp.log()).abs()[maskoe].mean()
		# 	return flow2*4, flow3*8,flow4*16,flow5*32,flow6*64,loss, dchange2[:,0], iexp2[:,0]

		# else:
		# 	return flow2,  dchange2[0,0], iexp2[0,0]


class Depth_Decoder_SelfMono(nn.Module):
	def __init__(self):
		super(Depth_Decoder_SelfMono, self).__init__()

		self.conv6 = conv_no_bn(32, 1, isReLU=False)
		self.conv5 = conv_no_bn(32, 1, isReLU=False)
		self.conv4 = conv_no_bn(32, 1, isReLU=False)
		self.conv3 = conv_no_bn(32, 1, isReLU=False)

	def forward(self, upfeats):
		disp6 = self.conv6(upfeats[0])
		disp5 = self.conv5(upfeats[1])
		disp4 = self.conv5(upfeats[2])
		disp3 = self.conv5(upfeats[3])
		disp2 = self.conv5(upfeats[4])

		return [disp6, disp5, disp4, disp3, disp2]


#class Depth_Decoder_ResNet_Style(nn.Module):
#	def __init__(self, ch_in, ch_out):
#		super(Depth_Decoder_ResNet_Style,self).__init__()

#		self.conv0 = 

class Disp_Decoder_Skip_Connection(nn.Module):
	def __init__(self,ch_in):
		super(Disp_Decoder_Skip_Connection, self).__init__()

		self.upconvs = nn.ModuleList()
		self.ch_in = ch_in[::-1]
		self.ch_dec = [16, 32, 64, 96, 128, 196, 196]
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
			x = [F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)]
			if i != 5:
				x += [input_features[i+1]]
			x = torch.cat(x,1)
			x = self.upconvs[i][1](x)

			disps.append(self.sigmoid(self.disp_decoders[i](x))*0.3)

		return disps