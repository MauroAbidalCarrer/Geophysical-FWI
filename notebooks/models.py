from itertools import pairwise, accumulate

import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import max_pool2d, interpolate

from config import CHANNELS_DIMENSION


class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, start_features:int, depth:int, dataset_stats:dict[str, float]={}):
        super().__init__()
        # TODO: Check that this will be saved when serializing the weights to a pth file
        self.dataset_stats = dataset_stats
        # Define the channels per depth of the Unet
        chs_per_depth = [start_features * 2 ** i for i in range(depth)]
        # Instantiate the Unet
        down_blocks_chns_it = pairwise([in_channels, *chs_per_depth])
        self.down_blocks = [ResidualBlock(in_chs, out_chs) for in_chs, out_chs in down_blocks_chns_it]
        self.down_blocks = nn.ModuleList(self.down_blocks)
        # Instantiate the bottle neck
        self.bottle_neck_block = ResidualBlock(chs_per_depth[-1], chs_per_depth[-1])
        # Instantialte the downblocks
        self.up_blocks = nn.ModuleList()
        for in_chs, out_chs in pairwise([*chs_per_depth[::-1], out_channels]):
            self.up_blocks.append(ResidualBlock(in_chs * 2, out_chs))
        self.last_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x:Tensor) -> Tensor:
        # normed_x = (x - self.dataset_stats.get("x_mean", 0)) / self.dataset_stats.get("x_std", 1)
        encoder_outputs = list(accumulate(self.down_blocks, encode, initial=x))
        out = self.bottle_neck_block(encoder_outputs[-1])
        for up_block, encode_output in zip(self.up_blocks, encoder_outputs[::-1]):
            out = decode(out, encode_output, up_block)
        out = self.last_conv(out)
        # scaled_out = out * self.dataset_stats.get("y_std", 1) + self.dataset_stats.get("y_mean", 0)
        return out

class ResidualBlock(nn.Module):
    """2 Convulution block + residual"""
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity() 
        else:
            # May want to set bias to False ?
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x:Tensor) -> Tensor:
        return self.skip_connection(x) + self.blocks(x)

class ConvBlock(nn.Sequential):
    """3x3+1Padding Conv, BN, ReLu"""
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

def encode(x:Tensor, module:nn.Module) -> Tensor:
    return max_pool2d(module(x), 2)

def decode(prev_block_x:Tensor, skip_x:Tensor, module:nn.Module) -> Tensor:
    x_diff = skip_x.shape[2] - prev_block_x.shape[2]
    y_diff = skip_x.shape[3] - prev_block_x.shape[3]
    # todo: Center padding?
    padded_prev_block_x = nn.functional.pad(prev_block_x, (x_diff, 0, y_diff, 0))
    x = torch.concatenate((padded_prev_block_x, skip_x), CHANNELS_DIMENSION)
    out = module(x)
    return interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)