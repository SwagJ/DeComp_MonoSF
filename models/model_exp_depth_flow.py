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
from .pwc import pwc_dc_net


class Flow_ExpDepth(nn.Module):
    def __init__(self, args):
        super(Flow_ExpDepth, self).__init__()

        self._args = args
        self.pwc_net = pwc_dc_net('./pretrained_pwc/pwc_net.pth.tar')
        self.depth_decoder = 

    def forward(self,input_dict):
