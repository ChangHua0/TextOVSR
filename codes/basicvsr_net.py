import sys
sys.path.insert(0, '/data/ch/Opera_Restoration/RealWorldVSR/RealBasicVSR-master/CLIP/')  # 例如 /home/ch/project/CLIP
import clip as clip
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer, make_layer_norm)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

# add CLIP
from transformers import CLIPTokenizer, CLIPTextModel,CLIPImageProcessor,CLIPVisionModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import SlicedAttnProcessor

# NegVSR
from einops.layers.torch import Rearrange
import random
import os
import numpy as np
from PIL import Image
 

@BACKBONES.register_module()
class BasicVSRNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ResidualBlocksWithInputConv_norm(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # main.append(nn.InstanceNorm2d(out_channels, affine=True))
        main.append(nn.LayerNorm(out_channels))

        # residual blocks
        main.append(
            make_layer_norm(
                ResidualBlockNoBN, num_blocks, out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)



# 复现NegVSR
print("read FLIR noise Sequence begin")
noise_dir = '/codes/collect_noise_sequence/Demos/Noise_Sequences/'


folderNames = os.listdir(noise_dir)
noiseSequences = []
for folderName in folderNames:
    images = []
    imgNames = os.listdir(os.path.join(noise_dir,folderName))
    imgNames = sorted(imgNames)
    for imgName in imgNames:
        img = Image.open(os.path.join(noise_dir,folderName, imgName))
        img_tensor = torch.from_numpy((np.array(img) / 255.0)).permute(2, 0, 1).float().unsqueeze(0).cuda()
        images.append(img_tensor)
    noiseSequence = torch.stack(images, dim=1)
    noiseSequences.append(noiseSequence)
print("read FLIR noise Sequence end noiseSequences_len = ",len(noiseSequences))
# print("read opera noise Sequence end noiseSequences_len = ",len(noiseSequences))
#read noise Sequence


@BACKBONES.register_module()
class NegVSRNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        #propagation branches
        # self.backward_resblocks = ResidualBlocksWithInputConv(
        #     mid_channels + 3, mid_channels, num_blocks)
        # self.forward_resblocks = ResidualBlocksWithInputConv(
        #     mid_channels + 3, mid_channels, num_blocks)

        
        #syx propagation branches
        # self.backward_resblocks_layer1 = ResidualBlocksWithInputConv(
        #      mid_channels + 3, mid_channels, 1)
        # self.forward_resblocks_layer1  = ResidualBlocksWithInputConv(
        #      mid_channels + 3, mid_channels, 1)
        self.backward_resblocks_layer10 = ResidualBlocksWithInputConv(
           mid_channels + 3, mid_channels, 10)
        self.forward_resblocks_layer10 = ResidualBlocksWithInputConv(
           mid_channels + 3, mid_channels, 10)
        self.to_patch = Rearrange('b c (h1 h) (w1 w)  -> (b h1 w1) c h w ', h1=16, w1=16)
        self.to_entire = Rearrange('(b h1 w1) c h w  -> b c (h1 h) (w1 w) ', h1=16, w1=16)
        #syx propagation branches
        
        #SYX-begin
        # self.carafe1=CARAFEPack(channels=64,scale_factor=2).cuda()
        # self.carafe2=CARAFEPack(channels=64,scale_factor=2).cuda()
        #self.conv_3to64 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=1).requires_grad_(False)
        #SYX-end

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)
        
        # compute optical flow
        #flows_forward, flows_backward = self.compute_flow(lrs)

        # #noise sequence
        # # randomSeed = random.randint(0, 4083)
        # # noises = noiseSequences[randomSeed].cuda()
        # lrs_mixup_noiseSequences = []
        # # lrs_mixup_noiseSequences_for = []
        # # lrs_mixup_noiseSequences_back = []
        # for i in range(0,t):
        #
        #     #lrs_mixup_noiseSequence = lrs[:, i, :, :, :] * 0.9 + noises[:, i, :, :, :] * 0.1
        #     lrs_mixup_noiseSequence = lrs[:, i, :, :, :]
        #
        #     lrs_mixup_noiseSequence_for, lrs_mixup_noiseSequence_back = mask3(lrs_mixup_noiseSequence)
        #
        #     lrs_mixup_noiseSequences_for.append(lrs_mixup_noiseSequence_for)
        #     lrs_mixup_noiseSequences_back.append(lrs_mixup_noiseSequence_back)
        #
        # lrs_mixup_noiseSequences_for = torch.stack(lrs_mixup_noiseSequences_for, dim=1).cuda()
        # lrs_mixup_noiseSequences_back = torch.stack(lrs_mixup_noiseSequences_back, dim=1).cuda()
        # # noise sequence

        # # backward-time propagation
        # outputs = []
        # feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        # for i in range(t - 1, -1, -1):
        #     if i < t - 1:  # no warping required for the last timestep
        #         flow = flows_backward[:, i, :, :, :]
        #         feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
        #
        #     feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)# [1, 67, 64, 64])
        #     #feat_prop = torch.cat([lrs_mask[:, i, :, :, :], feat_prop], dim=1) #mask
        #     #feat_prop = self.edge_TM_backword(feat_prop)# edge_tm
        #     #feat_prop = torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1) #mixup_noise
        #     #feat_prop = self.edge_TM_backword(torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1))  # edge_tm+noise
        #     #feat_prop = torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1)#edge_tm2
        #     #feat_prop = torch.cat([lrs_mixupNoise_mask_back[:, i, :, :, :], feat_prop], dim=1)  # noise+mask
        #     #feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)#noiseSequence
        #
        #     #feat_prop = self.backward_resblocks(feat_prop)
        #     #feat_prop = self.backward_resblocks_layer1(blur(feat_prop))#1layer_blur
        #     feat_prop = self.backward_resblocks_layer1(feat_prop)# layer1
        #     #feat_prop = self.backward_resblocks_10(feat_prop)  # layer10
        #
        #
        #     outputs.append(feat_prop)
        # outputs = outputs[::-1]
        #
        # # forward-time propagation and upsampling
        # feat_prop = torch.zeros_like(feat_prop)
        # for i in range(0, t):
        #     lr_curr = lrs[:, i, :, :, :]
        #
        #     if i > 0:  # no warping required for the first timestep
        #         if flows_forward is not None:
        #             flow = flows_forward[:, i - 1, :, :, :]
        #         else:
        #             flow = flows_backward[:, -i, :, :, :]
        #         feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
        #
        #     feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
        #     #feat_prop = torch.cat([lrs_mask[:, i, :, :, :], feat_prop], dim=1) #mark
        #     #feat_prop = self.edge_TM_forword(feat_prop)  # edge_tm
        #     #feat_prop = torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1)  # mixup_noise
        #     #feat_prop = self.edge_TM_forword(torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1))  # edge_tm+noise
        #     #feat_prop = torch.cat([lrs_mixupNoise[:, i, :, :, :], feat_prop], dim=1)# edge_tm2
        #     #feat_prop = torch.cat([lrs_mixupNoise_mask_for[:, i, :, :, :], feat_prop], dim=1)  # noise+mask
        #     #feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence
        #
        #     #feat_prop = self.forward_resblocks(feat_prop)
        #     #feat_prop = self.forward_resblocks_layer1(blur(feat_prop))#layer1_blur
        #     feat_prop = self.forward_resblocks_layer1(feat_prop) # layer1
        #     ##feat_prop = self.forward_resblocks_10(feat_prop)  # layer10
        #
        #     # upsampling given the backward and forward features
        #     out = torch.cat([outputs[i], feat_prop], dim=1)
        #
        #     out = self.lrelu(self.fusion(out))           #in_c=128,out_c=64
        #     # out = self.lrelu(self.upsample1(out))        #in_c=64,out_c=64
        #     # out = self.lrelu(self.upsample2(out))        #in_c=64,out_c=64
        #
        #     #syx_upsample
        #     # noise_temp = self.conv_3to64(noises[:, i, :, :, :])
        #     # out = self.lrelu(self.carafe1(out,noise_temp)) #in_c=64,out_c=64
        #     # out = self.lrelu(self.carafe2(out,out))        #in_c=64,out_c=64
        #     out = self.lrelu(self.carafe1(out))        #in_c=64,out_c=64
        #     out = self.lrelu(self.carafe2(out))        #in_c=64,out_c=64
        #     # syx_upsample
        #
        #     out = self.lrelu(self.conv_hr(out))          #in_c=64,out_c=64
        #     out = self.conv_last(out)                    #in_c=64,out_c=3
        #     base = self.img_upsample(lr_curr)            #[1,3,256,256]
        #     out += base
        #     outputs[i] = out


        if self.training:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            lrs_mixup_noiseSequences_mixupRot = []
            for i in range(0,t):
                lrs_mixup_noiseSequence_mixupRot = lrs[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5       #mixup+rot
                lrs_mixup_noiseSequences_mixupRot.append(lrs_mixup_noiseSequence_mixupRot)                      #mixup+rot
            lrs_mixup_noiseSequences_mixupRot = torch.stack(lrs_mixup_noiseSequences_mixupRot, dim=1).cuda()    #mixup+rot
            # noise sequence

            # #lrs_rot = self.NegRot(lrs)# rot+mixup
            # lrs_rot = self.NegRot(lrs_mixup_noiseSequences_mixupRot)# mixup+rot

            #rot_pro = (torch.randint(25,35,(1,)).cuda())/100
            rot_pro = (torch.randint(0,10,(1,)).cuda())/10
            lrs_rot = self.rot_p(lrs_mixup_noiseSequences_mixupRot,rot_pro)  # mixup+rot

            #outputs_rot = self.propagation(lrs_rot, isMixup=True)  # rot+mixup

            outputs_rot = self.propagation(lrs_rot,isMixup=False)   # mixup+rot

        #outputs_rot = self.propagation(self.NegRot(lrs), isMixup=True)  # input & noise Rot mixup

        # # rot noise with random p
        # rot_pro=(torch.round(torch.rand(1)*10)/10).cuda()
        # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
        # noises = noiseSequences[randomSeed].cuda()
        # noises_rot = self.rot_p(noises,rot_pro)
        # outputs = self.propagation_rotNoise(lrs,noises_rot, isMixup=True)
        # # rot noise with random p
        outputs = self.propagation(lrs,isMixup=False)
        # return torch.stack(outputs_rot, dim=1) #only negmix
        if self.training:
            return torch.stack(outputs, dim=1), torch.stack(outputs_rot, dim=1)
        else:
            return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
        
    def NegRot(self,images):
        images = images.squeeze(0)
        patchs = self.to_patch(images) #[B,3,8,8]
        b, _, _, _ = patchs.shape
        patchs_ro = torch.zeros_like(patchs)
        for i in range(b):
            k = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
            patchs_ro[i] = torch.rot90(patchs[i], k=k, dims=[1, 2])
        patchs_entire = self.to_entire(patchs_ro) #[b,3,64,64]
        return patchs_entire.unsqueeze(0)
    
    def rot_p(self,images,p):
        images = images.squeeze(0) #[30,3,64,64]
        patchs = self.to_patch(images) #[B,3,8,8]
        b, _, _, _ = patchs.shape
        patchs_ro = torch.zeros_like(patchs)
        for i in range(b):
            randomSeed = torch.rand(1).cuda()
            if randomSeed<=p:
                k = torch.randint(low=0,high=5,size=(1,)).cuda().item()
                patchs_ro[i] = torch.rot90(patchs[i], k=k, dims=[1, 2])
            else:
                patchs_ro[i] = patchs[i]
        patchs_entire = self.to_entire(patchs_ro) #[b,3,64,64]
        return patchs_entire.unsqueeze(0)
    
    def propagation(self,lrs_temp,isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            #noises = self.NegRot(noises) # input & noise Rot mixup
            lrs_mixup_noiseSequences = []
            # lrs_mixup_noiseSequences_for = []
            # lrs_mixup_noiseSequences_back = []
            for i in range(0,t):

                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5
                #lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :]
                # lrs_mixup_noiseSequence_for, lrs_mixup_noiseSequence_back = mask3(lrs_mixup_noiseSequence)
                # lrs_mixup_noiseSequences_for.append(lrs_mixup_noiseSequence_for)
                # lrs_mixup_noiseSequences_back.append(lrs_mixup_noiseSequence_back)
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)

            # lrs_mixup_noiseSequences_for = torch.stack(lrs_mixup_noiseSequences_for, dim=1).cuda()
            # lrs_mixup_noiseSequences_back = torch.stack(lrs_mixup_noiseSequences_back, dim=1).cuda()
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()
            # noise sequence
        else:
            lrs_mixup_noiseSequences=lrs_temp
        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.backward_resblocks_layer10(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.forward_resblocks_layer10(feat_prop) 
            
            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs

    def propagation_rotNoise(self, lrs_temp,noises_rot, isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        lrs_mixup_noiseSequences = []
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            for i in range(0, t):
                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises_rot[:, i, :, :, :] * 0.5
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()

        else:
            lrs_mixup_noiseSequences = lrs_temp

        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            # feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence
            feat_prop = self.backward_resblocks_layer1(feat_prop)  # layer1
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            # feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence
            feat_prop = self.forward_resblocks_layer1(feat_prop)  # layer1

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs


@BACKBONES.register_module()
class TextOVSRNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None,clip_model_name: str = "openai/clip-vit-large-patch14-336",
                 transformer_layers_per_block: int = 1,
                num_attention_heads = 1,
                resnet_groups: int = 32,
                use_linear_projection: bool = False,
                only_cross_attention: bool = False,
                # use_linear_projection: bool = True,
                # only_cross_attention: bool = True,
                upcast_attention: bool = False,
                attention_type: str = "default",):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        #propagation branches
        # self.backward_resblocks = ResidualBlocksWithInputConv(
        #     mid_channels + 3, mid_channels, num_blocks)
        # self.forward_resblocks = ResidualBlocksWithInputConv(
        #     mid_channels + 3, mid_channels, num_blocks)

        
        #syx propagation branches
        # self.backward_resblocks_layer1 = ResidualBlocksWithInputConv(
        #      mid_channels + 3, mid_channels, 1)
        # self.forward_resblocks_layer1  = ResidualBlocksWithInputConv(
        #      mid_channels + 3, mid_channels, 1)


        self.backward_resblocks_layer10 = ResidualBlocksWithInputConv(
           mid_channels + 3, mid_channels, 10)
        self.forward_resblocks_layer10 = ResidualBlocksWithInputConv(
           mid_channels + 3, mid_channels, 10)
        self.backward_resblocks_layer10_po = ResidualBlocksWithInputConv(
           mid_channels * 2, mid_channels, 10)
        self.forward_resblocks_layer10_po = ResidualBlocksWithInputConv(
           mid_channels * 2, mid_channels, 10)
        self.to_patch = Rearrange('b c (h1 h) (w1 w)  -> (b h1 w1) c h w ', h1=16, w1=16)
        self.to_entire = Rearrange('(b h1 w1) c h w  -> b c (h1 h) (w1 w) ', h1=16, w1=16)
        #syx propagation branches
        
        #SYX-begin
        # self.carafe1=CARAFEPack(channels=64,scale_factor=2).cuda()
        # self.carafe2=CARAFEPack(channels=64,scale_factor=2).cuda()
        #self.conv_3to64 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=1).requires_grad_(False)
        #SYX-end

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # CLIP 文本编码器（冻结所有参数）
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # —— 新增：把 CLIP 的 768 维投到 mid_channels (64)
        hidden_size = self.text_encoder.config.hidden_size  # == 768
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, mid_channels),   # 512→64
        )
        
        # # 在 backward 和 forward attention 前各自加一个 LayerNorm
        # # 这里用 LayerNorm 并对 “通道” 维度做归一化
        # self.backward_attn_norm = nn.LayerNorm(self.mid_channels)
        # self.forward_attn_norm  = nn.LayerNorm(self.mid_channels)

        # Transformer2D cross-attention modules for backward and forward propagation
        self.backward_cross_attn = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=mid_channels // num_attention_heads,
            in_channels=mid_channels,
            num_layers=transformer_layers_per_block,
            cross_attention_dim=mid_channels,
            norm_num_groups=resnet_groups,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type
        )
        self.forward_cross_attn = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=mid_channels // num_attention_heads,
            in_channels=mid_channels,
            num_layers=transformer_layers_per_block,
            cross_attention_dim=mid_channels,
            norm_num_groups=resnet_groups,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            attention_type=attention_type)

        self.frame_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, prompts, captions):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            prompts: list of strings, length B*T, each corresponding to each frame.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        if self.training:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            lrs_mixup_noiseSequences_mixupRot = []
            for i in range(0,t):
                lrs_mixup_noiseSequence_mixupRot = lrs[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5       #mixup+rot
                lrs_mixup_noiseSequences_mixupRot.append(lrs_mixup_noiseSequence_mixupRot)                      #mixup+rot
            lrs_mixup_noiseSequences_mixupRot = torch.stack(lrs_mixup_noiseSequences_mixupRot, dim=1).cuda()    #mixup+rot
            # noise sequence

            # #lrs_rot = self.NegRot(lrs)# rot+mixup
            # lrs_rot = self.NegRot(lrs_mixup_noiseSequences_mixupRot)# mixup+rot

            #rot_pro = (torch.randint(25,35,(1,)).cuda())/100
            rot_pro = (torch.randint(0,10,(1,)).cuda())/10
            lrs_rot = self.rot_p(lrs_mixup_noiseSequences_mixupRot,rot_pro)  # mixup+rot

            #outputs_rot = self.propagation(lrs_rot, isMixup=True)  # rot+mixup

            # outputs_rot = self.propagation(lrs_rot,isMixup=False)   # mixup+rot
            outputs_rot = self.propagation_clip_ne(lrs_rot,prompts,isMixup=False)   # mixup+rot

        #outputs_rot = self.propagation(self.NegRot(lrs), isMixup=True)  # input & noise Rot mixup

        # # rot noise with random p
        # rot_pro=(torch.round(torch.rand(1)*10)/10).cuda()
        # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
        # noises = noiseSequences[randomSeed].cuda()
        # noises_rot = self.rot_p(noises,rot_pro)
        # outputs = self.propagation_rotNoise(lrs,noises_rot, isMixup=True)
        # # rot noise with random p
        # outputs = self.propagation_clip_po(lrs,captions,isMixup=False)
        outputs,sent_emb,words_embs = self.propagation_clip_po(lrs,captions,isMixup=False)
        # return torch.stack(outputs_rot, dim=1) #only negmix
        if self.training:
            # return torch.stack(outputs, dim=1), torch.stack(outputs_rot, dim=1)
            return torch.stack(outputs, dim=1), torch.stack(outputs_rot, dim=1),sent_emb,words_embs
        else:
            return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
        
    def NegRot(self,images):
        images = images.squeeze(0)
        patchs = self.to_patch(images) #[B,3,8,8]
        b, _, _, _ = patchs.shape
        patchs_ro = torch.zeros_like(patchs)
        for i in range(b):
            k = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
            patchs_ro[i] = torch.rot90(patchs[i], k=k, dims=[1, 2])
        patchs_entire = self.to_entire(patchs_ro) #[b,3,64,64]
        return patchs_entire.unsqueeze(0)
    
    def rot_p(self,images,p):
        images = images.squeeze(0) #[30,3,64,64]
        patchs = self.to_patch(images) #[B,3,8,8]
        b, _, _, _ = patchs.shape
        patchs_ro = torch.zeros_like(patchs)
        for i in range(b):
            randomSeed = torch.rand(1).cuda()
            if randomSeed<=p:
                k = torch.randint(low=0,high=5,size=(1,)).cuda().item()
                patchs_ro[i] = torch.rot90(patchs[i], k=k, dims=[1, 2])
            else:
                patchs_ro[i] = patchs[i]
        patchs_entire = self.to_entire(patchs_ro) #[b,3,64,64]
        return patchs_entire.unsqueeze(0)
    
    def propagation(self,lrs_temp,isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            #noises = self.NegRot(noises) # input & noise Rot mixup
            lrs_mixup_noiseSequences = []
            # lrs_mixup_noiseSequences_for = []
            # lrs_mixup_noiseSequences_back = []
            for i in range(0,t):

                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5
                #lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :]
                # lrs_mixup_noiseSequence_for, lrs_mixup_noiseSequence_back = mask3(lrs_mixup_noiseSequence)
                # lrs_mixup_noiseSequences_for.append(lrs_mixup_noiseSequence_for)
                # lrs_mixup_noiseSequences_back.append(lrs_mixup_noiseSequence_back)
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)

            # lrs_mixup_noiseSequences_for = torch.stack(lrs_mixup_noiseSequences_for, dim=1).cuda()
            # lrs_mixup_noiseSequences_back = torch.stack(lrs_mixup_noiseSequences_back, dim=1).cuda()
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()
            # noise sequence
        else:
            lrs_mixup_noiseSequences=lrs_temp

        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.backward_resblocks_layer10(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.forward_resblocks_layer10(feat_prop) 
            
            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs

    def propagation_clip_ne(self,lrs_temp,prompts,isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            #noises = self.NegRot(noises) # input & noise Rot mixup
            lrs_mixup_noiseSequences = []
            # lrs_mixup_noiseSequences_for = []
            # lrs_mixup_noiseSequences_back = []
            for i in range(0,t):

                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5
                #lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :]
                # lrs_mixup_noiseSequence_for, lrs_mixup_noiseSequence_back = mask3(lrs_mixup_noiseSequence)
                # lrs_mixup_noiseSequences_for.append(lrs_mixup_noiseSequence_for)
                # lrs_mixup_noiseSequences_back.append(lrs_mixup_noiseSequence_back)
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)

            # lrs_mixup_noiseSequences_for = torch.stack(lrs_mixup_noiseSequences_for, dim=1).cuda()
            # lrs_mixup_noiseSequences_back = torch.stack(lrs_mixup_noiseSequences_back, dim=1).cuda()
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()
            # noise sequence
        else:
            lrs_mixup_noiseSequences=lrs_temp

        # --- 文本条件准备: 批量编码 B*T 条 caption ---
        # prompts: List[str] len=B*T
        text_inputs = self.tokenizer(
            prompts,
            padding='max_length',
            # max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_inputs = text_inputs.to(self.text_encoder.device)
        text_out = self.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        # [B*T, seq_len, hidden_size]
        token_embeds = text_out.last_hidden_state
        # 投影到 mid_channels: Linear applies on last dim
        token_embeds = self.text_proj(token_embeds)  # [B*T, seq_len, C]
        # reshape to [B, T, seq_len, C]
        token_embeds = token_embeds.view(n, t, token_embeds.size(1), self.mid_channels)


        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.backward_resblocks_layer10(feat_prop)

            # # —— Pre-Norm before cross-attn ——
            # # 1) 对 feat_prop 做 layernorm：需要先 permute
            # #    从 (B, C, H, W) → (B, H, W, C)
            # feat_norm = feat_prop.permute(0, 2, 3, 1)
            # feat_norm = self.backward_attn_norm(feat_norm)
            # #    再 permute 回 (B, C, H, W)
            # feat_norm = feat_norm.permute(0, 3, 1, 2)

            # 跨注意力：取第 i 帧的所有 token 条件
            txt_cond = token_embeds[:, i, :, :]  # [B, seq_len, C]
            attn_out = self.backward_cross_attn(
                hidden_states=feat_prop,
                encoder_hidden_states=txt_cond
            ).sample

            # 3) 残差连接
            feat_prop = feat_prop + attn_out

            outputs.append(feat_prop)
        outputs = outputs[::-1]
        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.forward_resblocks_layer10(feat_prop)

            # # —— Pre-Norm before forward cross-attn ——
            # feat_norm = feat_prop.permute(0, 2, 3, 1)
            # feat_norm = self.forward_attn_norm(feat_norm)
            # feat_norm = feat_norm.permute(0, 3, 1, 2)

            # 跨注意力：取第 i 帧的所有 token 条件
            txt_cond = token_embeds[:, i]  # [B, seq_len, C]
            attn_out = self.forward_cross_attn(
                hidden_states=feat_prop,
                encoder_hidden_states=txt_cond
            ).sample

            feat_prop = feat_prop + attn_out 
            
            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs

    def propagation_clip_po(self,lrs_temp,prompts,isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            #noise sequence
            # randomSeed = torch.randint(low=0,high=325,size=(1,)).cuda().item()
            randomSeed = torch.randint(low=0,high=39,size=(1,)).cuda().item()
            noises = noiseSequences[randomSeed].cuda()
            #noises = self.NegRot(noises) # input & noise Rot mixup
            lrs_mixup_noiseSequences = []
            # lrs_mixup_noiseSequences_for = []
            # lrs_mixup_noiseSequences_back = []
            for i in range(0,t):

                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises[:, i, :, :, :] * 0.5
                #lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :]
                # lrs_mixup_noiseSequence_for, lrs_mixup_noiseSequence_back = mask3(lrs_mixup_noiseSequence)
                # lrs_mixup_noiseSequences_for.append(lrs_mixup_noiseSequence_for)
                # lrs_mixup_noiseSequences_back.append(lrs_mixup_noiseSequence_back)
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)

            # lrs_mixup_noiseSequences_for = torch.stack(lrs_mixup_noiseSequences_for, dim=1).cuda()
            # lrs_mixup_noiseSequences_back = torch.stack(lrs_mixup_noiseSequences_back, dim=1).cuda()
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()
            # noise sequence
        else:
            lrs_mixup_noiseSequences=lrs_temp

        # --- 文本条件准备: 批量编码 B*T 条 caption ---
        # 提取Caption特征
        # prompts: List[str] len=B*T
        text_inputs = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_inputs = text_inputs.to(self.text_encoder.device)
        text_out = self.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        # [B*T, seq_len, hidden_size]
        token_embeds = text_out.last_hidden_state
        sent_emb = text_out.pooler_output
        # 投影到 mid_channels: Linear applies on last dim
        token_embeds = self.text_proj(token_embeds)  # [B*T, seq_len, C]
        # reshape to [B, T, seq_len, C]
        token_embeds = token_embeds.view(n, t, token_embeds.size(1), self.mid_channels)

        # 提取帧特征
        # lrs_mixup_noiseSequences 是五维张量
        B, T, C, H, W = lrs_mixup_noiseSequences.shape
        lrs_mixup_noiseSequences = lrs_mixup_noiseSequences.view(-1, C, H, W)  # [B*T, C, H, W]
        # 提取输入帧的特征便于后续融合文本特征
        frame_features = self.frame_extractor(lrs_mixup_noiseSequences)
        frame_features = frame_features.view(B, T, 64, H, W)

        # 融合Caption特征和帧特征
        # 在同样的 device 和 dtype 下分配一个空张量
        mixed_features = torch.zeros(B, T, 64, H, W, 
                            device=frame_features.device, 
                            dtype=frame_features.dtype)
        # 对每一帧融合文本图像特征
        # 跨注意力：取第 i 帧的所有 token 条件
        for i in range(0, t):
            frame_feature = frame_features[:, i, :, :, :]
            txt_cond = token_embeds[:, i, :, :]  # [B, T, seq_len, C]
            attn_out = self.backward_cross_attn(
                hidden_states=frame_feature,
                encoder_hidden_states=txt_cond
            ).sample
            mixed_feature = frame_feature + attn_out
            mixed_features[:, i] = mixed_feature

        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([mixed_features[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.backward_resblocks_layer10_po(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            #feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            #feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([mixed_features[:, i, :, :, :], feat_prop], dim=1)# noiseSequence
            feat_prop = self.forward_resblocks_layer10_po(feat_prop)            
            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs, sent_emb, token_embeds


    def propagation_rotNoise(self, lrs_temp,noises_rot, isMixup=False):
        n, t, c, h, w = lrs_temp.size()
        lrs_mixup_noiseSequences = []
        flows_forward, flows_backward = self.compute_flow(lrs_temp)
        if isMixup:
            for i in range(0, t):
                lrs_mixup_noiseSequence = lrs_temp[:, i, :, :, :] * 0.5 + noises_rot[:, i, :, :, :] * 0.5
                lrs_mixup_noiseSequences.append(lrs_mixup_noiseSequence)
            lrs_mixup_noiseSequences = torch.stack(lrs_mixup_noiseSequences, dim=1).cuda()

        else:
            lrs_mixup_noiseSequences = lrs_temp

        outputs = []
        feat_prop = lrs_temp.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # feat_prop = torch.cat([lrs_temp[:, i, :, :, :], feat_prop], dim=1)  # [1, 67, 64, 64])
            # feat_prop = torch.cat([lrs_mixup_noiseSequences_for[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence
            feat_prop = self.backward_resblocks_layer1(feat_prop)  # layer1
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_temp[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            # feat_prop = torch.cat([lrs_mixup_noiseSequences_back[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence mask
            feat_prop = torch.cat([lrs_mixup_noiseSequences[:, i, :, :, :], feat_prop], dim=1)  # noiseSequence
            feat_prop = self.forward_resblocks_layer1(feat_prop)  # layer1

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))  # in_c=128,out_c=64
            out = self.lrelu(self.upsample1(out))  # in_c=64,out_c=64
            out = self.lrelu(self.upsample2(out))  # in_c=64,out_c=64
            out = self.lrelu(self.conv_hr(out))  # in_c=64,out_c=64
            out = self.conv_last(out)  # in_c=64,out_c=3
            base = self.img_upsample(lr_curr)  # [1,3,256,256]
            out += base
            outputs[i] = out
        return outputs

