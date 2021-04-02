from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation
import numpy as np

from .modules_sceneflow import get_grid, WarpingLayer_SF, WarpingLayer_Flow_Exp, MonoSF_Disp_Exp_Decoder, WarpingLayer_Flow_Exp_Plus, WarpingLayer_Flow, WarpingLayer_SF_PWC
from .modules_sceneflow import initialize_msra, upsample_outputs_as, normalize_feature
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork,MonoFlow_Disp_Decoder, ContextNetwork_Flow_Disp, ContextNetwork_Flow_DispC_v1_1, ContextNetwork_Flow_DispC_v1_2
from .modules_sceneflow import Flow_Decoder, ContextNetwork_Flow, Disp_Decoder_Skip_Connection, Disp_Decoder, ContextNetwork_Disp, Feature_Decoder, ContextNetwork_Disp_DispC, Disp_DispC_Decoder
from .modules_sceneflow import Feature_Decoder_ppV1, Exp_Decoder_ppV1_Dense, ContextNetwork_Exp, WarpingLayer_Flow_Exp, affine, MonoFlow_DispC_Decoder_v1_1, MonoFlow_DispC_Decoder_v1_2
from .modules_sceneflow import MonoSF_DispC_Decoder, ContextNetwork_DispC, ContextNetwork_Disp_DispC_v2_2, Disp_DispC_Decoder_v2_2, WarpingLayer_SF_DispC, ContextNetwork_DispC_SF, DispC_Decoder

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing

import collections


class MonoSceneFlow(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSceneFlowDecoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(32 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        #print("feature pyramid shape:",x1_pyramid[0].shape, x1_pyramid[1].shape, x1_pyramid[2].shape, x1_pyramid[3].shape, x1_pyramid[4].shape,x1_pyramid[5].shape, x1_pyramid[6].shape)

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f, disp_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_res_b, disp_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        #print("top-level sf shape:", sceneflows_f[-1].shape)
        flows_f = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        #print("flow_f shape:",flows_f[0].shape, flows_f[1].shape, flows_f[2].shape, flows_f[3].shape, flows_f[4].shape)

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoSF_DispC(nn.Module):
    def __init__(self, args):
        super(MonoSF_DispC, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSF_DispC_Decoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork_DispC_SF(32 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        #print("feature pyramid shape:",x1_pyramid[0].shape, x1_pyramid[1].shape, x1_pyramid[2].shape, x1_pyramid[3].shape, x1_pyramid[4].shape,x1_pyramid[5].shape, x1_pyramid[6].shape)

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1, dispC_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2, dispC_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1, dispC_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1, dispC_f], dim=1))
                x2_out, flow_b_res, disp_l2, dispC_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2, dispC_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f, disp_l1, dispC_res_f = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, dispC_f], dim=1))
                flow_res_b, disp_l2, dispC_res_b = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, dispC_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        #print("top-level sf shape:", sceneflows_f[-1].shape)
        #flows_f = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        #print("flow_f shape:",flows_f[0].shape, flows_f[1].shape, flows_f[2].shape, flows_f[3].shape, flows_f[4].shape)

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1], x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict


class MonoSF_Disp_Exp(nn.Module):
    def __init__(self, args):
        super(MonoSF_Disp_Exp, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_Flow_Exp()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSF_Disp_Exp_Decoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(32 + 3 + 1,is_exp=True)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows_f = []
        flows_b = []
        disps_1 = []
        disps_2 = []
        exps_f = []
        exps_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                exp_f = interpolate2d_as(exp_f, x1, mode="bilinear")
                exp_b = interpolate2d_as(exp_b, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'], exp_f)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l1, k1, input_dict['aug_size'], exp_b)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1, exp_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2, exp_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1, exp_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1,exp_f], dim=1))
                x2_out, flow_b_res, disp_l2, exp_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2,exp_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                exp_f = self.sigmoid(exp_f) * 0.1
                exp_b = self.sigmoid(exp_b) * 0.1
                flows_f.append(flow_f)
                flows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                exps_f.append(exp_f)
                exps_b.append(exp_b)
            else:
                flow_res_f, disp_l1, exp_f = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, exp_f], dim=1))
                flow_res_b, disp_l2, exp_b = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, exp_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                flows_f.append(flow_f)
                flows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                exps_f.append(exp_f)
                exps_b.append(exp_b)                
                break


        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(flows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(flows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        output_dict['exp_f'] = upsample_outputs_as(exps_f[::-1],x1_rev)
        output_dict['exp_b'] = upsample_outputs_as(exps_b[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                output_dict_r['exp_f'][ii] = torch.flip(output_dict_r['exp_f'][ii], [3])
                output_dict_r['exp_b'][ii] = torch.flip(output_dict_r['exp_b'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            exp_f_pp = []
            exp_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                exp_f_pp.append(post_processing(output_dict['exp_f'][ii], torch.flip(output_dict_flip['exp_f'][ii], [3])))
                exp_b_pp.append(post_processing(output_dict['exp_b'][ii], torch.flip(output_dict_flip['exp_b'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['exp_f_pp'] = exp_f_pp
            output_dict['exp_b_pp'] = exp_b_pp

        return output_dict

class MonoSF_Disp_Exp_Plus(nn.Module):
    def __init__(self, args):
        super(MonoSF_Disp_Exp_Plus, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_Flow_Exp_Plus()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSF_Disp_Exp_Decoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(32 + 3 + 1,is_exp=True)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows_f = []
        flows_b = []
        disps_1 = []
        disps_2 = []
        exps_f = []
        exps_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                exp_f = interpolate2d_as(exp_f, x1, mode="bilinear")
                exp_b = interpolate2d_as(exp_b, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'], exp_f)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l1, k1, input_dict['aug_size'], exp_b)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1, exp_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2, exp_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1, exp_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1,exp_f], dim=1))
                x2_out, flow_b_res, disp_l2, exp_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2,exp_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                exp_f = self.sigmoid(exp_f) * 0.1
                exp_b = self.sigmoid(exp_b) * 0.1
                flows_f.append(flow_f)
                flows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                exps_f.append(exp_f)
                exps_b.append(exp_b)
            else:
                flow_res_f, disp_l1, exp_f = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, exp_f], dim=1))
                flow_res_b, disp_l2, exp_b = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, exp_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                flows_f.append(flow_f)
                flows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                exps_f.append(exp_f)
                exps_b.append(exp_b)                
                break


        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(flows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(flows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        output_dict['exp_f'] = upsample_outputs_as(exps_f[::-1],x1_rev)
        output_dict['exp_b'] = upsample_outputs_as(exps_b[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                output_dict_r['exp_f'][ii] = torch.flip(output_dict_r['exp_f'][ii], [3])
                output_dict_r['exp_b'][ii] = torch.flip(output_dict_r['exp_b'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            exp_f_pp = []
            exp_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                exp_f_pp.append(post_processing(output_dict['exp_f'][ii], torch.flip(output_dict_flip['exp_f'][ii], [3])))
                exp_b_pp.append(post_processing(output_dict['exp_b'][ii], torch.flip(output_dict_flip['exp_b'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['exp_f_pp'] = exp_f_pp
            output_dict['exp_b_pp'] = exp_b_pp

        return output_dict



class MonoSceneFlow_Disp_Res(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow_Disp_Res, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSceneFlowDecoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(32 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1_res = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x2_out, flow_b_res, disp_l2_res = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level and l != 0:
                disp_l1_res = self.sigmoid(disp_l1_res) * 0.3
                disp_l2_res = self.sigmoid(disp_l2_res) * 0.3
                disp_l1 = disp_l1 + disp_l1_res
                disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            elif l != self.output_level and l == 0:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f, disp_l1_res = self.context_networks(torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_res_b, disp_l2_res = self.context_networks(torch.cat([x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = disp_l1 + disp_l1_res
                disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoFlow_Disp(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 2 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoFlow_Disp_Decoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork_Flow_Disp(32 + 2 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_flow(x2, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_flow(x1, flow_b)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f, disp_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_res_b, disp_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict

class MonoFlow_Disp_Seperate_NoWarp(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Seperate_NoWarp, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 2
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = Flow_Decoder(num_ch_in)   
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork_Flow(32 + 2)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        #disps_1 = []
        #disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                #disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                #disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_flow(x2, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_flow(x1, flow_b)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f], dim=1))
                x2_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                #disp_l1 = self.sigmoid(disp_l1) * 0.3
                #disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                #disps_1.append(disp_l1)
                #disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks(torch.cat([x1_out, flow_f], dim=1))
                flow_res_b = self.context_networks(torch.cat([x2_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                #disps_1.append(disp_l1)
                #disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        #output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        #output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        disp_l1 = self.disp_estimator1(x1_pyramid)
        disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        output_dict['disp_l1'] = disp_l1[::-1][:5]
        output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoFlow_Disp_Seperate_Warp_OG_Decoder(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Seperate_Warp_OG_Decoder, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                disp_l1 = disp_l1 + disp_l1_res
                disp_l2 = disp_l2 + disp_l2_res

            #x1_feats.append(x1_out)
            #x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1_res = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2_res = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                disp_l1 = disp_l1 + disp_l1_res
                disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []
        corr_f = []
        corr_b = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1_flow)
            x2_feats.append(x2_flow)
            corr_f.append(out_corr_relu_flow_f)
            corr_b.append(out_corr_relu_flow_b)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        if backbone_mode == False:
            output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        else:
            output_dict['flow_f'] = sceneflows_f[::-1]
            output_dict['flow_b'] = sceneflows_b[::-1]
            output_dict['disp_l1'] = disps_1[::-1]
            output_dict['disp_l2'] = disps_2[::-1]
            output_dict['x1_feats'] = x1_feats[::-1]
            output_dict['x2_feats'] = x2_feats[::-1]
            output_dict['x1_rev'] = x1_rev
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        output_dict['x1_feats'] = x1_feats
        output_dict['x2_feats'] = x2_feats
        output_dict['corr_f'] = corr_f
        output_dict['corr_b'] = corr_b
        output_dict['x1_rev'] = x1_rev
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if (self.training or self._args.exp_training) or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.exp_training:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            output_dict['output_dict_flip'] = output_dict_flip

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Seperate_Warp_OG_Decoder_Feat_Norm, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            x1_flow_norm, x2_warp_flow_norm = normalize_feature([x1_flow, x2_warp_flow])
            x2_flow_norm, x1_warp_flow_norm = normalize_feature([x2_flow, x1_warp_flow])
            out_corr_flow_f = Correlation.apply(x1_flow_norm, x2_warp_flow_norm, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow_norm, x1_warp_flow_norm, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            x1_disp_norm, x2_warp_disp_norm = normalize_feature([x1_disp, x2_warp_disp])
            x2_disp_norm, x1_warp_disp_norm = normalize_feature([x2_disp, x1_warp_disp])
            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        if backbone_mode == False:
            output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        else:
            output_dict['flow_f'] = sceneflows_f[::-1]
            output_dict['flow_b'] = sceneflows_b[::-1]
            output_dict['disp_l1'] = disps_1[::-1]
            output_dict['disp_l2'] = disps_2[::-1]
            output_dict['x1_feats'] = x1_feats[::-1]
            output_dict['x2_feats'] = x2_feats[::-1]
            output_dict['x1_rev'] = x1_rev
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if (self.training or self._args.exp_training) or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.exp_training:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict



class MonoSceneFlow_PWC(nn.Module):
    def __init__(self, args):
        super(MonoSceneFlow_PWC, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF_PWC()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_sf = MonoSceneFlowDecoder(num_ch_in)            
            self.flow_estimators.append(layer_sf)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = ContextNetwork(32 + 3 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        #print("feature pyramid shape:",x1_pyramid[0].shape, x1_pyramid[1].shape, x1_pyramid[2].shape, x1_pyramid[3].shape, x1_pyramid[4].shape,x1_pyramid[5].shape, x1_pyramid[6].shape)

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'], l)  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'], l)

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                #flow_f = flow_f + flow_f_res
                #flow_b = flow_b + flow_b_res

            x1_feats.append(x1_out)
            x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_f, disp_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1], dim=1))
                flow_b, disp_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2], dim=1))
                #flow_f = flow_f + flow_res_f
                #flow_b = flow_b + flow_res_b
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        #print("top-level sf shape:", sceneflows_f[-1].shape)
        flows_f = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        #print("flow_f shape:",flows_f[0].shape, flows_f[1].shape, flows_f[2].shape, flows_f[3].shape, flows_f[4].shape)

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict



#########################################################
#
# Proposed v1: Flow + flow_x as coefficient for disp change.
#              flow_x as multiplier.
#              Warpping with flow
#              Disp using warpped feature. 
#              flow_x decoder as expansion style: bottom is 
#              pyramid pooling, upper levels are skip connector
#
#########################################################
class MonoFlowExp_ppV1(nn.Module):
    def __init__(self, args):
        super(MonoFlowExp_ppV1, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.upconv_layers_exp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()
        self.exp_estimators = nn.ModuleList()
        #self.flow_x_estimator = Flow_x_Decoder_ppV1()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
                num_ch_in_exp = self.dim_corr + ch
                is_bottom = True
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                num_ch_in_exp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                self.upconv_layers_exp.append(upconv(32, 32, 3, 2))
                is_bottom = False
                #self.disp_decoder.append
            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            layer_exp = Exp_Decoder_ppV1_Dense(num_ch_in_exp, is_bottom)
            self.feature_decoder.append(Feature_Decoder_ppV1(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)
            self.exp_estimators.append(layer_exp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.context_networks_exp = ContextNetwork_Exp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # for (ii,item) in enumerate(x1_pyramid):
        #     print("idx: %d Shape: %d , %d"%(ii,item.size()[2],item.size()[3]))

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        expansion_f = []
        expansion_b = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x2_warp_exp = x2_exp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
                x1_warp_exp = x1_exp
            else:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                exp_f = interpolate2d_as(exp_f, x1_exp, mode="bilinear")
                exp_b = interpolate2d_as(exp_b, x1_exp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)
                x1_exp_out = self.upconv_layers_exp[l-1](x1_exp_out)
                x2_exp_out = self.upconv_layers_exp[l-1](x2_exp_out)
                x2_warp_exp = self.warping_layer_flow(x2_exp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_exp = self.warping_layer_flow(x1_exp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            out_corr_exp_f = Correlation.apply(x1_exp, x2_warp_exp, self.corr_params)
            out_corr_exp_b = Correlation.apply(x2_exp, x1_warp_exp, self.corr_params)
            out_corr_relu_exp_f = self.leakyRELU(out_corr_exp_f)
            out_corr_relu_exp_b = self.leakyRELU(out_corr_exp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                x1_exp_out, exp_f = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp], dim=1))
                x2_exp_out, exp_b = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                x1_exp_out, exp_f_res = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp, x1_exp_out, exp_f], dim=1))
                x2_exp_out, exp_b_res = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp, x2_exp_out, exp_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                exp_f = exp_f + exp_f_res
                exp_f = exp_b + exp_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            #x1_feats.append(x1_out)
            #x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                exp_res_f = self.context_networks_exp(torch.cat([x1_exp_out, exp_f], dim=1))
                exp_res_b = self.context_networks_exp(torch.cat([x2_exp_out, exp_b], dim=1))
                exp_f = exp_f + exp_res_f
                exp_b = exp_b + exp_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['exp_f'] = upsample_outputs_as(expansion_f[::-1],x1_rev)
        output_dict['exp_b'] = upsample_outputs_as(expansion_b[::-1],x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            exp_f_pp = []
            exp_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                exp_f_pp.append(post_processing(output_dict['exp_f'][ii], flow_horizontal_flip(output_dict_flip['exp_f'][ii])))
                exp_b_pp.append(post_processing(output_dict['exp_b'][ii], flow_horizontal_flip(output_dict_flip['exp_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['exp_f_pp'] = exp_f_pp
            output_dict['exp_b_pp'] = exp_b_pp

        return output_dict


#########################################################
#
# Proposed v2: Flow + flow_x as coefficient for disp change.
#              flow_x as multiplier.
#              Warpping with flow and sceneflow
#              Disp using warpped feature. 
#              flow_x decoder as expansion style: bottom is 
#              pyramid pooling, upper levels are skip connector
#
#########################################################
class MonoFlowExp_ppV1_2(nn.Module):
    def __init__(self, args):
        super(MonoFlowExp_ppV1_2, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        self.warping_layer_flow_exp = WarpingLayer_Flow_Exp()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.upconv_layers_exp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()
        self.exp_estimators = nn.ModuleList()
        #self.flow_x_estimator = Flow_x_Decoder_ppV1()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
                num_ch_in_exp = self.dim_corr + ch
                is_bottom = True
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                num_ch_in_exp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                self.upconv_layers_exp.append(upconv(32, 32, 3, 2))
                is_bottom = False
                #self.disp_decoder.append
            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            layer_exp = Exp_Decoder_ppV1_Dense(num_ch_in_exp, is_bottom)
            self.feature_decoder.append(Feature_Decoder_ppV1(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)
            self.exp_estimators.append(layer_exp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.context_networks_exp = ContextNetwork_Exp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # for (ii,item) in enumerate(x1_pyramid):
        #     print("idx: %d Shape: %d , %d"%(ii,item.size()[2],item.size()[3]))

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        expansion_f = []
        expansion_b = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x2_warp_exp = x2_exp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
                x1_warp_exp = x1_exp
            else:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                exp_f = interpolate2d_as(exp_f, x1_exp, mode="bilinear")
                exp_b = interpolate2d_as(exp_b, x1_exp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)
                x1_exp_out = self.upconv_layers_exp[l-1](x1_exp_out)
                x2_exp_out = self.upconv_layers_exp[l-1](x2_exp_out)
                x2_warp_exp = self.warping_layer_flow_exp(x2_exp, flow_f, exp_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp_exp = self.warping_layer_flow_exp(x1_exp, flow_b, exp_b, disp_l2, k2, input_dict['aug_size'])
                #sceneflow, disp, k1, input_size

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            out_corr_exp_f = Correlation.apply(x1_exp, x2_warp_exp, self.corr_params)
            out_corr_exp_b = Correlation.apply(x2_exp, x1_warp_exp, self.corr_params)
            out_corr_relu_exp_f = self.leakyRELU(out_corr_exp_f)
            out_corr_relu_exp_b = self.leakyRELU(out_corr_exp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                x1_exp_out, exp_f = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp], dim=1))
                x2_exp_out, exp_b = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                x1_exp_out, exp_f_res = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp, x1_exp_out, exp_f], dim=1))
                x2_exp_out, exp_b_res = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp, x2_exp_out, exp_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                exp_f = exp_f + exp_f_res
                exp_f = exp_b + exp_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            #x1_feats.append(x1_out)
            #x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                exp_res_f = self.context_networks_exp(torch.cat([x1_exp_out, exp_f], dim=1))
                exp_res_b = self.context_networks_exp(torch.cat([x2_exp_out, exp_b], dim=1))
                exp_f = exp_f + exp_res_f
                exp_b = exp_b + exp_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['exp_f'] = upsample_outputs_as(expansion_f[::-1],x1_rev)
        output_dict['exp_b'] = upsample_outputs_as(expansion_b[::-1],x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            exp_f_pp = []
            exp_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                exp_f_pp.append(post_processing(output_dict['exp_f'][ii], flow_horizontal_flip(output_dict_flip['exp_f'][ii])))
                exp_b_pp.append(post_processing(output_dict['exp_b'][ii], flow_horizontal_flip(output_dict_flip['exp_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['exp_f_pp'] = exp_f_pp
            output_dict['exp_b_pp'] = exp_b_pp

        return output_dict

##
## add affine matrix and use optical expansion as input
##
##
class MonoFlowExp_ppV1_3(nn.Module):
    def __init__(self, args):
        super(MonoFlowExp_ppV1_3, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.upconv_layers_exp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()
        self.exp_estimators = nn.ModuleList()
        #self.flow_x_estimator = Flow_x_Decoder_ppV1()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
                num_ch_in_exp = self.dim_corr + ch + 2
                is_bottom = True
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                num_ch_in_exp = self.dim_corr + ch + 32 + 1 + 2
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                self.upconv_layers_exp.append(upconv(32, 32, 3, 2))
                is_bottom = False
                #self.disp_decoder.append
            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            layer_exp = Exp_Decoder_ppV1_Dense(num_ch_in_exp, is_bottom)
            self.feature_decoder.append(Feature_Decoder_ppV1(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)
            self.exp_estimators.append(layer_exp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.context_networks_exp = ContextNetwork_Exp(32 + 1 + 2)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        # for (ii,item) in enumerate(x1_pyramid):
        #     print("idx: %d Shape: %d , %d"%(ii,item.size()[2],item.size()[3]))

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        expansion_f = []
        expansion_b = []
        expmask_f = []
        expmask_b = []


        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x2_warp_exp = x2_exp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
                x1_warp_exp = x1_exp
            else:
                x1_disp,x1_flow,x1_exp = self.feature_decoder[l](x1)
                x2_disp,x2_flow,x2_exp = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                exp_f = interpolate2d_as(exp_f, x1_exp, mode="bilinear")
                exp_b = interpolate2d_as(exp_b, x1_exp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)
                x1_exp_out = self.upconv_layers_exp[l-1](x1_exp_out)
                x2_exp_out = self.upconv_layers_exp[l-1](x2_exp_out)
                x2_warp_exp = self.warping_layer_flow(x2_exp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_exp = self.warping_layer_flow(x1_exp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            out_corr_exp_f = Correlation.apply(x1_exp, x2_warp_exp, self.corr_params)
            out_corr_exp_b = Correlation.apply(x2_exp, x1_warp_exp, self.corr_params)
            out_corr_relu_exp_f = self.leakyRELU(out_corr_exp_f)
            out_corr_relu_exp_b = self.leakyRELU(out_corr_exp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                exp_if, err_f, mask_f = affine(flow_f.detach(), pw=1)
                exp_ib, err_b, mask_b = affine(flow_b.detach(), pw=1)
                x1_exp_out, exp_f = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp, exp_if, err_f], dim=1))
                x2_exp_out, exp_b = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp, exp_if, err_f], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                exp_if, err_f, mask_f = affine(flow_f.detach(), pw=1)
                exp_ib, err_b, mask_b = affine(flow_b.detach(), pw=1)
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                x1_exp_out, exp_f = self.exp_estimators[l](torch.cat([out_corr_relu_exp_f, x1_exp, x1_exp_out, exp_f, exp_if, err_f], dim=1))
                x2_exp_out, exp_b = self.exp_estimators[l](torch.cat([out_corr_relu_exp_b, x2_exp, x2_exp_out, exp_b, exp_ib, err_b], dim=1))
                #exp_f = exp_f + exp_f_res
                #exp_f = exp_b + exp_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            #x1_feats.append(x1_out)
            #x2_feats.append(x2_out)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b)
                expmask_f.append(mask_f)
                expmask_b.append(mask_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                exp_if, err_f, mask_f = affine(flow_f.detach(), pw=1)
                exp_ib, err_b, mask_b = affine(flow_b.detach(), pw=1)
                exp_f = self.context_networks_exp(torch.cat([x1_exp_out, exp_f, exp_if, err_f], dim=1))
                exp_b = self.context_networks_exp(torch.cat([x2_exp_out, exp_b, exp_ib, err_b], dim=1))
                #exp_f = exp_f + exp_res_f
                #exp_b = exp_b + exp_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                expansion_f.append(exp_f)
                expansion_b.append(exp_b) 
                expmask_f.append(mask_f)
                expmask_b.append(mask_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['exp_f'] = upsample_outputs_as(expansion_f[::-1],x1_rev)
        output_dict['exp_b'] = upsample_outputs_as(expansion_b[::-1],x1_rev)
        #output_dict['mask_f'] = upsample_outputs_as(expmask_f[::-1],x1_rev)
        #output_dict['mask_b'] = upsample_outputs_as(expmask_b[::-1],x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.sf_sup:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            exp_f_pp = []
            exp_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                exp_f_pp.append(post_processing(output_dict['exp_f'][ii], flow_horizontal_flip(output_dict_flip['exp_f'][ii])))
                exp_b_pp.append(post_processing(output_dict['exp_b'][ii], flow_horizontal_flip(output_dict_flip['exp_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['exp_f_pp'] = exp_f_pp
            output_dict['exp_b_pp'] = exp_b_pp

        return output_dict



class MonoFlow_DispC_v1_1(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v1_1, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 3
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = MonoFlow_DispC_Decoder_v1_1(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow_DispC_v1_1(32 + 3)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f, dispC_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b, dispC_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res, dispC_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f, dispC_f], dim=1))
                x2_flow_out, flow_b_res, dispC_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b, dispC_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f, dispC_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f, dispC_f], dim=1))
                flow_res_b, dispC_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b, dispC_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict

class MonoFlow_DispC_v1_2(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v1_2, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 3
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = MonoFlow_DispC_Decoder_v1_2(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow_DispC_v1_2(32 + 3)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f, dispC_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b, dispC_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res, dispC_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f, dispC_f], dim=1))
                x2_flow_out, flow_b_res, dispC_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b, dispC_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f, dispC_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f, dispC_f], dim=1))
                flow_res_b, dispC_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b, dispC_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict

class MonoFlow_DispC_v2_1(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v2_1, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 2
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_DispC_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp_DispC(32 + 2)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1, dispC_f = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2, dispC_b = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1, dispC_f_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1, dispC_f], dim=1))
                x2_disp_out, disp_l2, dispC_b_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2, dispC_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1, dispC_res_f = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1, dispC_f], dim=1))
                disp_l2, dispC_res_b = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict


class MonoFlow_DispC_v2_2(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v2_2, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 2
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_DispC_Decoder_v2_2(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp_DispC_v2_2(32 + 2)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1, dispC_f = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2, dispC_b = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1, dispC_f_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1, dispC_f], dim=1))
                x2_disp_out, disp_l2, dispC_b_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2, dispC_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1, dispC_res_f = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1, dispC_f], dim=1))
                disp_l2, dispC_res_b = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict

#############################################################
# dispC warpping with 3d information
#
#############################################################
class MonoFlow_DispC_v2_3(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v2_3, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_SF_DispC()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 2
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_DispC_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp_DispC(32 + 2)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f, dispC_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b, dispC_b, disp_l2, k2, input_dict['aug_size'])
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f, dispC_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b, dispC_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1, dispC_f = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2, dispC_b = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1, dispC_f_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1, dispC_f], dim=1))
                x2_disp_out, disp_l2, dispC_b_res = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2, dispC_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1, dispC_res_f = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1, dispC_f], dim=1))
                disp_l2, dispC_res_b = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict


class MonoFlow_Disp_Teacher(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Teacher, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        if backbone_mode == False:
            output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        else:
            output_dict['flow_f'] = sceneflows_f[::-1]
            output_dict['flow_b'] = sceneflows_b[::-1]
            output_dict['disp_l1'] = disps_1[::-1]
            output_dict['disp_l2'] = disps_2[::-1]
            output_dict['x1_feats'] = x1_feats[::-1]
            output_dict['x2_feats'] = x2_feats[::-1]
            output_dict['x1_rev'] = x1_rev
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l1_aug'], input_dict['input_k_l1'], input_dict['input_k_l2'], self._args.backbone_mode)

        input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
        input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
        k_l1_flip = input_dict["input_k_l1_flip_aug"]
        k_l2_flip = input_dict["input_k_l2_flip_aug"]

        output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

        input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
        input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
        k_r1_flip = input_dict["input_k_r1_flip_aug"]
        k_r2_flip = input_dict["input_k_r2_flip_aug"]

        output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)
        output_dict['output_dict_r'] = output_dict_r

        flow_f_pp = []
        flow_b_pp = []
        disp_l1_pp = []
        disp_l2_pp = []

        for ii in range(0, len(output_dict_flip['flow_f'])):

            flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
            flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
            disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
            disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

        output_dict['flow_f_pp'] = flow_f_pp
        output_dict['flow_b_pp'] = flow_b_pp
        output_dict['disp_l1_pp'] = disp_l1_pp
        output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict

class MonoFlow_Disp_Student(nn.Module):
    def __init__(self, args):
        super(MonoFlow_Disp_Student, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        if backbone_mode == False:
            output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
            output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
            output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
            output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        else:
            output_dict['flow_f'] = sceneflows_f[::-1]
            output_dict['flow_b'] = sceneflows_b[::-1]
            output_dict['disp_l1'] = disps_1[::-1]
            output_dict['disp_l2'] = disps_2[::-1]
            output_dict['x1_feats'] = x1_feats[::-1]
            output_dict['x2_feats'] = x2_feats[::-1]
            output_dict['x1_rev'] = x1_rev
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug_student'], input_dict['input_l2_aug_student'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if (self.training or self._args.exp_training) or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug_student'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug_student'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning or self._args.exp_training:

            input_l1_flip = torch.flip(input_dict['input_l1_aug_student'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug_student'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class MonoFlowDisp_Teacher_Student(nn.Module):
    def __init__(self, args):
        super(MonoFlowDisp_Teacher_Student, self).__init__()
        self._args = args
        self._teacher = MonoFlow_Disp_Teacher(args)
        self._student = MonoFlow_Disp_Student(args)
        if self._args.evaluation == False:
            state_dict = torch.load(self._args.backbone_weight)
            renamed_state_dict = collections.OrderedDict()
            for k,v in state_dict['state_dict'].items():
                name = k[7:]
                renamed_state_dict[name] = v

            self._teacher.load_state_dict(renamed_state_dict)
            self._student.load_state_dict(renamed_state_dict) 

    def forward(self, input_dict):
        output_dict = {}
        # cropping for student
        if (not self._args.evaluation) == True:
            self.eval()
            torch.set_grad_enabled(False)

            output_dict['teacher_dict'] = self._teacher(input_dict)
        
            torch.set_grad_enabled(True)          
            self.train()

        output_dict['student_dict'] = self._student(input_dict)

        return output_dict


class MonoFlow_DispC_v2_4(nn.Module):
    def __init__(self, args):
        super(MonoFlow_DispC_v2_4, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.flow_estimators = nn.ModuleList()
        #self.disp_estimator1 = Disp_Decoder_Skip_Connection(self.num_chs)
        #self.disp_estimator2 = Disp_Decoder_Skip_Connection(self.num_chs)
        self.upconv_layers_flow = nn.ModuleList()
        self.upconv_layers_disp = nn.ModuleList()
        self.upconv_layers_dispC = nn.ModuleList()
        self.feature_decoder = nn.ModuleList()
        self.disp_estimators = nn.ModuleList()
        self.dispC_estimators = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_flow = self.dim_corr + ch
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_flow = self.dim_corr + ch + 32 + 2
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_flow.append(upconv(32, 32, 3, 2))
                self.upconv_layers_disp.append(upconv(32, 32, 3, 2))
                self.upconv_layers_dispC.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append

            layer_sf = Flow_Decoder(num_ch_in_flow)
            layer_disp = Disp_Decoder(num_ch_in_disp)
            layer_dispC = DispC_Decoder(num_ch_in_disp)
            self.feature_decoder.append(Feature_Decoder(ch)) 
            #layer_disp = Disp_Decoder(num_ch_in)         
            self.flow_estimators.append(layer_sf)
            self.disp_estimators.append(layer_disp) 
            self.dispC_estimators.append(layer_dispC)           

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks_flow = ContextNetwork_Flow(32 + 2)
        self.context_networks_disp = ContextNetwork_Disp(32 + 1)
        self.context_networks_dispC = ContextNetwork_DispC(32 + 1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2, backbone_mode=False):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        disps_1 = []
        disps_2 = []
        dispCs_f = []
        dispCs_b = []
        x1_feats = []
        x2_feats = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                x2_warp_flow = x2_flow
                x2_warp_disp = x2_disp
                x1_warp_flow = x1_flow
                x1_warp_disp = x1_disp
            else:
                x1_disp,x1_flow = self.feature_decoder[l](x1)
                x2_disp,x2_flow = self.feature_decoder[l](x2)
                flow_f = interpolate2d_as(flow_f, x1_flow, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1_flow, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1_disp, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1_disp, mode="bilinear")
                dispC_f = interpolate2d_as(dispC_f, x1_disp, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_disp, mode="bilinear")
                x1_flow_out = self.upconv_layers_flow[l-1](x1_flow_out)
                x2_flow_out = self.upconv_layers_flow[l-1](x2_flow_out)
                x2_warp_flow = self.warping_layer_flow(x2_flow, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_flow = self.warping_layer_flow(x1_flow, flow_b)
                x1_disp_out = self.upconv_layers_disp[l-1](x1_disp_out)
                x2_disp_out = self.upconv_layers_disp[l-1](x2_disp_out)
                x2_warp_disp = self.warping_layer_flow(x2_disp, flow_f)  # becuase K can be changing when doing augmentation
                x1_warp_disp = self.warping_layer_flow(x1_disp, flow_b)
                x1_dispC_out = self.upconv_layers_dispC[l-1](x1_dispC_out)
                x2_dispC_out = self.upconv_layers_dispC[l-1](x1_dispC_out)

            # correlation
            out_corr_flow_f = Correlation.apply(x1_flow, x2_warp_flow, self.corr_params)
            out_corr_flow_b = Correlation.apply(x2_flow, x1_warp_flow, self.corr_params)
            out_corr_relu_flow_f = self.leakyRELU(out_corr_flow_f)
            out_corr_relu_flow_b = self.leakyRELU(out_corr_flow_b)

            out_corr_disp_f = Correlation.apply(x1_disp, x2_warp_disp, self.corr_params)
            out_corr_disp_b = Correlation.apply(x2_disp, x1_warp_disp, self.corr_params)
            out_corr_relu_disp_f = self.leakyRELU(out_corr_disp_f)
            out_corr_relu_disp_b = self.leakyRELU(out_corr_disp_b)

            # monosf estimator
            if l == 0:
                x1_flow_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_flow_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp], dim=1))
                x1_dispC_out, dispC_f = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_dispC_out, dispC_b = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                #print("out dims:",out_corr_relu_f.shape,x1.shape,x1_out.shape,flow_f.shape,disp_l1.shape)
                x1_flow_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_flow_out, flow_f], dim=1))
                x2_flow_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow, x2_flow_out, flow_b], dim=1))
                x1_disp_out, disp_l1 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_f, x1_disp, x1_disp_out, disp_l1], dim=1))
                x2_disp_out, disp_l2 = self.disp_estimators[l](torch.cat([out_corr_relu_disp_b, x2_disp, x2_disp_out, disp_l2], dim=1))
                x1_dispC_out, dispC_f_res = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_dispC_out, dispC_f], dim=1))
                x2_dispC_out, dispC_b_res = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x2_flow, x2_dispC_out, dispC_b], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res

            x1_feats.append(x1)
            x2_feats.append(x2)
            #print(flow_f.shape)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                flow_res_f = self.context_networks_flow(torch.cat([x1_flow_out, flow_f], dim=1))
                flow_res_b = self.context_networks_flow(torch.cat([x2_flow_out, flow_b], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                disp_l1 = self.context_networks_disp(torch.cat([x1_disp_out, disp_l1], dim=1))
                disp_l2 = self.context_networks_disp(torch.cat([x2_disp_out, disp_l2], dim=1))

                dispC_res_f = self.context_networks_dispC(torch.cat([x1_dispC_out, dispC_f], dim=1))
                dispC_res_b = self.context_networks_dispC(torch.cat([x2_dispC_out, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                #disp_l1 = disp_l1 + disp_l1_res
                #disp_l2 = disp_l2 + disp_l2_res
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2) 
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break


        x1_rev = x1_pyramid[::-1]
        x2_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['dispC_f'] = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        output_dict['dispC_b'] = upsample_outputs_as(dispCs_b[::-1], x1_rev)
        #output_dict['x1_feats'] = upsample_outputs_as(x1_feats[::-1],x1_rev)
        #output_dict['x2_feats'] = upsample_outputs_as(x2_feats[::-1],x1_rev)
        #disp_l1 = self.disp_estimator1(x1_pyramid)
        #disp_l2 = self.disp_estimator2(x2_pyramid)
        #print("disp shape:",disp_l1[1].shape, disp_l1[2].shape, disp_l1[3].shape, disp_l1[4].shape, disp_l1[5].shape)

        #output_dict['disp_l1'] = disp_l1[::-1][:5]
        #output_dict['disp_l2'] = disp_l2[::-1][:5]
        
        return output_dict


    def forward(self, input_dict):
        with open('%s/iter_counts.txt'%(self._args.save), 'r') as f:
            cur_iter = int(f.read())
        f.close()
        if cur_iter < self._args.start:
            for dispC_estimator in self.dispC_estimators:
                dispC_estimator.requires_grad_(False)
            self.context_networks_dispC.requires_grad_(False)
        else:
            for dispC_estimator in self.dispC_estimators:
                dispC_estimator.requires_grad_(True)
            self.context_networks_dispC.requires_grad_(True)
        #print(self.context_networks_dispC.conv_sf[0].weight.grad)

        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'], self._args.backbone_mode)

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip,self._args.backbone_mode)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip, self._args.backbone_mode)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict


class MonoFlowDisp_DispC(nn.Module):
    def __init__(self, args):
        super(MonoFlowDisp_DispC, self).__init__()
        self._args = args
        self._backbone = MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res(args)
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.output_level = 4
        self.num_levels = 7
        self.dim_corr = 81

        self.dispC_estimators = nn.ModuleList()
        self.upconv_layers_dispC = nn.ModuleList()
        if self._args.evaluation == False:
            state_dict = torch.load(self._args.backbone_weight)
            renamed_state_dict = collections.OrderedDict()
            for k,v in state_dict['state_dict'].items():
                name = k[7:]
                renamed_state_dict[name] = v

            self._backbone.load_state_dict(renamed_state_dict) 

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_dispC.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append
            layer_dispC = DispC_Decoder(num_ch_in_disp)
            self.dispC_estimators.append(layer_dispC)

        self.context_networks_dispC = ContextNetwork_DispC(32 + 1)

    def run_decoder(self, x1_feats, x2_feats, corr_f, corr_b, x1_rev):
        dispCs_f = []
        dispCs_b = []
        for l, (x1, x2) in enumerate(zip(x1_feats, x2_feats)):
            x1_flow = x1
            x2_flow = x2
            if l != 0:
                dispC_f = interpolate2d_as(dispC_f, x1_flow, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_flow, mode="bilinear")
                x1_dispC_out = self.upconv_layers_dispC[l-1](x1_dispC_out)
                x2_dispC_out = self.upconv_layers_dispC[l-1](x2_dispC_out)

            out_corr_relu_flow_f = corr_f[l]
            out_corr_relu_flow_b = corr_b[l]

            # monosf estimator
            if l == 0:
                x1_dispC_out, dispC_f = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_dispC_out, dispC_b = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                x1_dispC_out, dispC_f_res = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_dispC_out, dispC_f], dim=1))
                x2_dispC_out, dispC_b_res = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x2_flow, x1_dispC_out, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_f_res
                dispC_b = dispC_b + dispC_b_res
                

            # upsampling or post-processing
            if l != self.output_level:
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                dispC_res_f = self.context_networks_dispC(torch.cat([x1_dispC_out, dispC_f], dim=1))
                dispC_res_b = self.context_networks_dispC(torch.cat([x2_dispC_out, dispC_b], dim=1))
                dispC_f = dispC_f + dispC_res_f
                dispC_b = dispC_b + dispC_res_b
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break

        dispCs_f = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        dispCs_b = upsample_outputs_as(dispCs_b[::-1], x1_rev)

        return dispCs_f, dispCs_b

    def forward(self, input_dict):
        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        if (not self._args.evaluation) == True:
            self.eval()
            torch.set_grad_enabled(False)

            output_dict = self._backbone(input_dict)
        
            torch.set_grad_enabled(True)          
            self.train()
            output_dict_r = output_dict['output_dict_r']
        else:
            output_dict = self._backbone(input_dict)
            output_dict_flip = output_dict['output_dict_flip']

        output_dict['dispC_f'], output_dict['dispC_b'] = self.run_decoder(output_dict['x1_feats'], output_dict['x2_feats'], output_dict['corr_f'], output_dict['corr_b'], output_dict['x1_rev'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            output_dict_r['dispC_f'], output_dict_r['dispC_b'] = self.run_decoder(output_dict_r['x1_feats'], output_dict_r['x2_feats'], output_dict_r['corr_f'], output_dict_r['corr_b'], output_dict_r['x1_rev'])

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip['flow_f'], output_dict_flip['flow_b'] = self.run_decoder(output_dict_flip['x1_feats'], output_dict_flip['x2_feats'], output_dict_flip['corr_f'], output_dict_flip['corr_b'], output_dict_flip['x1_rev'])
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict


class MonoFlowDisp_Exp(nn.Module):
    def __init__(self, args):
        super(MonoFlowDisp_Exp, self).__init__()
        self._args = args
        self._backbone = MonoFlow_Disp_Seperate_Warp_OG_Decoder_No_Res(args)
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.output_level = 4
        self.num_levels = 7
        self.dim_corr = 81

        self.dispC_estimators = nn.ModuleList()
        self.upconv_layers_dispC = nn.ModuleList()
        if self._args.evaluation == False:
            state_dict = torch.load(self._args.backbone_weight)
            renamed_state_dict = collections.OrderedDict()
            for k,v in state_dict['state_dict'].items():
                name = k[7:]
                renamed_state_dict[name] = v

            self._backbone.load_state_dict(renamed_state_dict) 

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in_disp = self.dim_corr + ch 
            else:
                num_ch_in_disp = self.dim_corr + ch + 32 + 1
                self.upconv_layers_dispC.append(upconv(32, 32, 3, 2))
                #self.disp_decoder.append
            layer_dispC = DispC_Decoder(num_ch_in_disp)
            self.dispC_estimators.append(layer_dispC)

        self.context_networks_dispC = ContextNetwork_DispC(32 + 1)

    def run_decoder(self, x1_feats, x2_feats, corr_f, corr_b, x1_rev):
        dispCs_f = []
        dispCs_b = []
        for l, (x1, x2) in enumerate(zip(x1_feats, x2_feats)):
            x1_flow = x1
            x2_flow = x2
            if l != 0:
                dispC_f = interpolate2d_as(dispC_f, x1_flow, mode="bilinear")
                dispC_b = interpolate2d_as(dispC_b, x1_flow, mode="bilinear")
                x1_dispC_out = self.upconv_layers_dispC[l-1](x1_dispC_out)
                x2_dispC_out = self.upconv_layers_dispC[l-1](x2_dispC_out)

            out_corr_relu_flow_f = corr_f[l]
            out_corr_relu_flow_b = corr_b[l]

            # monosf estimator
            if l == 0:
                x1_dispC_out, dispC_f = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow], dim=1))
                x2_dispC_out, dispC_b = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_b, x2_flow], dim=1))
                #print("bottom layer dim:",x1_out.shape,flow_f.shape,disp_l1.shape)
            else:
                x1_dispC_out, dispC_f = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x1_flow, x1_dispC_out, dispC_f], dim=1))
                x2_dispC_out, dispC_b = self.dispC_estimators[l](torch.cat([out_corr_relu_flow_f, x2_flow, x1_dispC_out, dispC_b], dim=1))
                

            # upsampling or post-processing
            if l != self.output_level:
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)
            else:
                dispC_res_f = self.context_networks_dispC(torch.cat([x1_dispC_out, dispC_f], dim=1))
                dispC_res_b = self.context_networks_dispC(torch.cat([x2_dispC_out, dispC_b], dim=1))
                dispCs_f.append(dispC_f)
                dispCs_b.append(dispC_b)               
                break

        dispCs_f = upsample_outputs_as(dispCs_f[::-1], x1_rev)
        dispCs_b = upsample_outputs_as(dispCs_b[::-1], x1_rev)

        return dispCs_f, dispCs_b

    def forward(self, input_dict):
        output_dict = {}

        ## Left
        #print("input image size:",input_dict['input_l1_aug'].shape)
        #print("input image size:",input_dict['input_l2_aug'].shape)
        if (not self._args.evaluation) == True:
            self.eval()
            torch.set_grad_enabled(False)

            output_dict = self._backbone(input_dict)
        
            torch.set_grad_enabled(True)          
            self.train()
            output_dict_r = output_dict['output_dict_r']
        else:
            output_dict = self._backbone(input_dict)
            output_dict_flip = output_dict['output_dict_flip']

        output_dict['dispC_f'], output_dict['dispC_b'] = self.run_decoder(output_dict['x1_feats'], output_dict['x2_feats'], output_dict['corr_f'], output_dict['corr_b'], output_dict['x1_rev'])

        #print("Training:", self.training)
        #print("Evaluation:", self._args.evaluation)
        #print("SF_Sup:", self._args.sf_sup)
        
        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.finetuning and not self._args.evaluation):
            output_dict_r['dispC_f'], output_dict_r['dispC_b'] = self.run_decoder(output_dict_r['x1_feats'], output_dict_r['x2_feats'], output_dict_r['corr_f'], output_dict_r['corr_b'], output_dict_r['x1_rev'])

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['dispC_f'][ii] = flow_horizontal_flip(output_dict_r['dispC_f'][ii])
                output_dict_r['dispC_b'][ii] = flow_horizontal_flip(output_dict_r['dispC_b'][ii])
                #output_dict_r['x1_feats'][ii] = torch.flip(output_dict_r['x1_feats'][ii], [3])
                #output_dict_r['x2_feats'][ii] = torch.flip(output_dict_r['x2_feats'][ii], [3])
                #print("output_dict_r[disp_l2] size:", output_dict_r['disp_l1'][ii].size())

            output_dict['output_dict_r'] = output_dict_r
            #print("generating right output dict")

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation or self._args.finetuning:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip['flow_f'], output_dict_flip['flow_b'] = self.run_decoder(output_dict_flip['x1_feats'], output_dict_flip['x2_feats'], output_dict_flip['corr_f'], output_dict_flip['corr_b'], output_dict_flip['x1_rev'])
            dispC_f_pp = []
            dispC_b_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):
                dispC_f_pp.append(post_processing(output_dict['dispC_f'][ii], flow_horizontal_flip(output_dict_flip['dispC_f'][ii])))
                dispC_b_pp.append(post_processing(output_dict['dispC_b'][ii], flow_horizontal_flip(output_dict_flip['dispC_b'][ii])))
            output_dict['dispC_f_pp'] = dispC_f_pp
            output_dict['dispC_b_pp'] = dispC_b_pp

        return output_dict