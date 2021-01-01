from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF, WarpingLayer_Flow_Exp, MonoSF_Disp_Exp_Decoder, WarpingLayer_Flow_Exp_Plus
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder, ContextNetwork

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing


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
