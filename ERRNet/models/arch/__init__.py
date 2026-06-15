# Add your custom network here
from .default import DRNet, DilatedRefiner
import torch.nn as nn


def _parse_dilations(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(',') if item.strip()]
    return list(value)


def basenet(in_channels, out_channels, **kwargs):
    kwargs['resblock_dilations'] = _parse_dilations(kwargs.get('resblock_dilations'))
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    kwargs['resblock_dilations'] = _parse_dilations(kwargs.get('resblock_dilations'))
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)
