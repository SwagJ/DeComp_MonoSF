from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging
import numpy as np

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF, WarpingLayer_Flow_Exp, MonoSF_Disp_Exp_Decoder, WarpingLayer_Flow_Exp_Plus
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing
from .modules_flow_expdepth import Expansion_Decoder, Disp_Decoder_Skip_Connection
from .pwc import pwc_dc_net
from .modules_flow_expdepth import get_grid


class Flow_Expansion(nn.Module):
	def __init__(self, args):
		super(Flow_Expansion, self).__init__()

		self.args = args
		self.pwc_net = pwc_dc_net('./models/pretrained_pwc/pwc_net_chairs.pth.tar')

		self.expansion = Expansion_Decoder(args,exp_unc=False)

	def forward(self,im0,im1):

		if self.training: # if only fine-tuning expansion 
			reset=True
			self.eval()
			torch.set_grad_enabled(False)
		else: reset=False

		corrs, flows, feats, feat_pyramid_1, feat_pyramid_2 = self.pwc_net(im0, im1)

		if reset: 
			torch.set_grad_enabled(True)          
			self.train()

		dchange, iexp = self.expansion(flows[-1]*20,feat_pyramid_1[-2], im0)

		return dchange, iexp, flows

class Mono_Expansion(nn.Module):
	def __init__(self, args):
		super(Mono_Expansion, self).__init__()

		self._args = args
		self.net = Flow_Expansion(args)
		self.maxdisp = 256
		self.fac = 1

	def affine_mask(self,pref,flow, pw=3):
		"""
		pref: reference coordinates
		pw: patch width
		"""
		flmask = flow[:,2:]
		flow = flow[:,:2]
		#print(flow.shape)
		#print(pref.shape)
		b,_,lh,lw=flow.shape
		ptar = pref + flow
		pref = tf.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
		ptar = tf.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
		
		conf_flow = flmask
		conf_flow = tf.unfold(conf_flow,(pw*2+1,pw*2+1), padding=(pw)).view(b,1,(pw*2+1)**2,lh,lw)
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

	def forward(self, input_dict):
		output_dict = {}
		# future variables
		imgAux_f = input_dict['imgAux_f'].cuda()
		flow_f = input_dict['flow_f'].cuda()
		# past variables
		#print(self._args.evaluation)
		#print("Input shape:", input_dict['im0_f'].shape)
		#forward pair
		dchange_f, iexp_f, flows_f = self.net(input_dict['im0_f'].cuda(), input_dict['im1_f'].cuda())
		b,_,h,w = flows_f[-1].shape
		output_dict['dchange_f'] = dchange_f
		output_dict['iexp_f'] = iexp_f
		output_dict['flows_f'] = flows_f
		#print(imgAux_f.shape)
		mask_f = (flow_f[:,2,:,:] == 1).float() * (flow_f[:,0,:,:].abs() < self.maxdisp).float() * (flow_f[:,1,:,:].abs() < (self.maxdisp//self.fac)).float()
		#print(mask_f.shape)
		mask_f = ((mask_f * (imgAux_f[:,0,:,:] < 100).float() * (imgAux_f[:,0,:,:] > 0.01).float()) > 0).unsqueeze(1)
		#print("mask_f shape:", mask_f.shape)

		# compute future losses
		gt_depth_f = imgAux_f[:,0,:,:].unsqueeze(1)
		#print("gt_depth_f shape:", gt_depth_f.shape)
		gt_f3d_f =  imgAux_f[:,4:7,:,:].clone()
		#print("gt_f3d_f shape:", gt_f3d_f.shape)
		
		gt_dchange_f = (1 + gt_f3d_f[:,2,:,:].unsqueeze(1) / gt_depth_f)
		maskdc_f = (gt_dchange_f < 2) & (gt_dchange_f > 0.5) & mask_f

		#print("gt_dchange_f shape:", gt_dchange_f.shape)
		#print("dchange_f shape:", dchange_f.shape)
		


		#print("flow shape:", flow_f.shape)
		gt_expi_f, gt_expi_err_f ,maskoe_f = self.affine_mask(get_grid(b,4*h,4*w)[:,0].permute(0,3,1,2).repeat(b,1,1,1), flow_f,pw=3)
		gt_exp_f = (1. / gt_expi_f[:,0]).unsqueeze(1)
		#print("gt_exp_f shape:", gt_exp_f.shape)
		#print("iexp shape:", iexp_f.shape)
		#print("maskdc_f shape:", maskdc_f.shape)
		#print("maskoe_f shape:", maskoe_f.shape)

		loss_dc_f =  0.1* (dchange_f-gt_dchange_f.log()).abs()[maskdc_f].mean()
		loss_iexp_f = 0.1* (iexp_f-gt_exp_f.log()).abs()[maskoe_f.unsqueeze(1)].mean()
		output_dict['loss_dc_f'] = loss_dc_f
		output_dict['loss_iexp_f'] = loss_iexp_f
		output_dict['mask_f'] = mask_f



		if (not self._args.evaluation) and (not self._args.finetuning):
			imgAux_b = input_dict['imgAux_b'].cuda()
			flow_b = input_dict['flow_b'].cuda()
			dchange_b, iexp_b, flows_b = self.net(input_dict['im0_b'].cuda(), input_dict['im1_b'].cuda())
			output_dict['dchange_b'] = dchange_b
			output_dict['iexp_b'] = iexp_b
			output_dict['flows_b'] = flows_b

			mask_b = (flow_b[:,2,:,:] == 1).float() * (flow_b[:,0,:,:].abs() < self.maxdisp).float() * (flow_b[:,1,:,:].abs() < (self.maxdisp//self.fac)).float()
			mask_b = ((mask_b * (imgAux_b[:,0,:,:] < 100).float() * (imgAux_b[:,0,:,:] > 0.01).float()) > 0).unsqueeze(1)

			# compute future losses
			gt_depth_b = imgAux_b[:,0,:,:].unsqueeze(1)
			gt_f3d_b =  imgAux_b[:,4:7,:,:].clone()
			gt_dchange_b = (1 + gt_f3d_b[:,2,:,:].unsqueeze(1) / gt_depth_b)
			maskdc_b = (gt_dchange_b < 2) & (gt_dchange_b > 0.5) & mask_b

			gt_expi_b, gt_expi_err_b ,maskoe_b = self.affine_mask(get_grid(b,4*h,4*w)[:,0].permute(0,3,1,2).repeat(b,1,1,1), flow_b,pw=3)
			gt_exp_b = (1. / gt_expi_b[:,0]).unsqueeze(1)
			#print("maskdc_f shape:", maskdc_b.shape)
			#print("maskoe_f shape:", maskoe_b.shape)

			loss_dc_b =  0.1* (dchange_b-gt_dchange_b.log()).abs()[maskdc_b].mean()
			loss_iexp_b = 0.1* (iexp_b-gt_exp_b.log()).abs()[maskoe_b.unsqueeze(1)].mean()
			output_dict['loss_dc_b'] = loss_dc_b
			output_dict['loss_iexp_b'] = loss_iexp_b
			output_dict['mask_b'] = mask_b

		#output_dict['input_dict'] = input_dict

		return output_dict

#class Flow_ExpDepth(nn.Module):
#    def __init__(self, args):
#        super(Flow_ExpDepth, self).__init__()

#        self._args = args
#        self.pwc_net = pwc_dc_net('./pretrained_pwc/pwc_net.pth.tar')
#        self.depth_decoder = args.depth_decoder()

#    def forward(self,input_dict):



class PWC_Disp(nn.Module):
	def __init__(self, args):
		super(PWC_Disp, self).__init__()

		self.args = args
		self.pwc_net = pwc_dc_net('./models/pretrained_pwc/pwc_net.pth.tar')
		self.ch_in = [16, 32, 64, 96, 128, 196]

		self.disp_l = Disp_Decoder_Skip_Connection(self.ch_in)
		self.disp_r = Disp_Decoder_Skip_Connection(self.ch_in)

	def forward(self,input_dict):
		output_dict = {}
		iml0 = input_dict['input_l1_aug']
		iml1 = input_dict['input_l2_aug']
		#print("training:", self.training)

		if self.training or (not self.args.evaluation): # if only fine-tuning expansion 
			reset=True
			train_flag = self.training
			self.eval()
			torch.set_grad_enabled(False)
		else: reset=False

		corrs_l, flows_l, feats_l, feat_pyramid_l0, feat_pyramid_l1 = self.pwc_net(iml0, iml1)
		#print("feat_pyramid grad:", feat_pyramid_l1[0].requires_grad, feat_pyramid_l1[1].requires_grad, feat_pyramid_l1[2].requires_grad,feat_pyramid_l1[3].requires_grad
		#					, feat_pyramid_l1[4].requires_grad, feat_pyramid_l1[5].requires_grad)

		if reset: 
			imr0 = input_dict['input_r1_aug']
			imr1 = input_dict['input_r2_aug']
			corrs_r, flows_r, feats_r, feat_pyramid_r0, feat_pyramid_r1 = self.pwc_net(imr0,imr1)
			output_dict['flows_r'] = flows_r

			if train_flag:
				torch.set_grad_enabled(True)          
				self.train()

			disp_r0 = self.disp_r(feat_pyramid_r0)
			disp_r1 = self.disp_r(feat_pyramid_r1)
			output_dict['disp_r0'] = disp_r0
			output_dict['disp_r1'] = disp_r1


		disp_l0 = self.disp_l(feat_pyramid_l0)
		disp_l1 = self.disp_l(feat_pyramid_l1)	

		output_dict['disp_l0'] = disp_l0
		output_dict['disp_l1'] = disp_l1

		output_dict['flows_l'] = flows_l
		#print("feat_pyramid grad:", feat_pyramid_r1[0].requires_grad, feat_pyramid_r1[1].requires_grad, feat_pyramid_r1[2].requires_grad,flows_l[3].requires_grad
		#					, feat_pyramid_l1[4].requires_grad, feat_pyramid_l1[5].requires_grad)


		#print("grad require:",disp_l0[0].requires_grad, disp_l1[0].requires_grad, disp_r0[0].requires_grad, disp_r1[0].requires_grad)


		return output_dict

class PWC_Disp_Unfreeze(nn.Module):
	def __init__(self, args):
		super(PWC_Disp_Unfreeze, self).__init__()

		self.args = args
		self.pwc_net = pwc_dc_net('./models/pretrained_pwc/pwc_net.pth.tar')
		self.ch_in = [16, 32, 64, 96, 128, 196]

		self.disp_l = Disp_Decoder_Skip_Connection(self.ch_in)
		self.disp_r = Disp_Decoder_Skip_Connection(self.ch_in)

	def forward(self,input_dict):
		output_dict = {}
		iml0 = input_dict['input_l1_aug']
		iml1 = input_dict['input_l2_aug']
		#print("training:", self.training)
		#print("feat_pyramid grad:", feat_pyramid_l1[0].requires_grad, feat_pyramid_l1[1].requires_grad, feat_pyramid_l1[2].requires_grad,feat_pyramid_l1[3].requires_grad
		#					, feat_pyramid_l1[4].requires_grad, feat_pyramid_l1[5].requires_grad)

		corrs_l, flows_l, feats_l, feat_pyramid_l0, feat_pyramid_l1 = self.pwc_net(iml0, iml1)

		if self.training or (not self.args.evaluation): 
			imr0 = input_dict['input_r1_aug']
			imr1 = input_dict['input_r2_aug']
			corrs_r, flows_r, feats_r, feat_pyramid_r0, feat_pyramid_r1 = self.pwc_net(imr0,imr1)
			output_dict['flows_r'] = flows_r

			disp_r0 = self.disp_r(feat_pyramid_r0)
			disp_r1 = self.disp_r(feat_pyramid_r1)
			output_dict['disp_r0'] = disp_r0
			output_dict['disp_r1'] = disp_r1


		disp_l0 = self.disp_l(feat_pyramid_l0)
		disp_l1 = self.disp_l(feat_pyramid_l1)	

		output_dict['disp_l0'] = disp_l0
		output_dict['disp_l1'] = disp_l1

		output_dict['flows_l'] = flows_l
		#print("feat_pyramid grad:", feat_pyramid_r1[0].requires_grad, feat_pyramid_r1[1].requires_grad, feat_pyramid_r1[2].requires_grad,flows_l[3].requires_grad
		#					, feat_pyramid_l1[4].requires_grad, feat_pyramid_l1[5].requires_grad)


		#print("grad require:",disp_l0[0].requires_grad, disp_l1[0].requires_grad, disp_r0[0].requires_grad, disp_r1[0].requires_grad)


		return output_dict
