import lightning as L
import torch
import torch.nn.functional as F
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel


def build_gaussian_kernel(kernel_size, sigma, device, dtype):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('Gaussian kernel size must be a positive odd integer')
    if sigma <= 0:
        raise ValueError('Gaussian sigma must be positive')

    coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return (kernel_2d / kernel_2d.sum()).to(dtype=dtype)


def gaussian_lowpass(img, kernel_size, sigma):
    pad = kernel_size // 2
    if img.size(-2) <= pad or img.size(-1) <= pad:
        raise ValueError(
            f'Input spatial size {tuple(img.shape[-2:])} is too small for '
            f'reflect padding with kernel_size={kernel_size}'
        )

    kernel = build_gaussian_kernel(
        kernel_size, sigma, device=img.device, dtype=img.dtype
    )
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(img.size(1), 1, 1, 1)
    padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(padded, kernel, groups=img.size(1))


def build_reflection_target(
    inp,
    target_t,
    target_r,
    mode='residual',
    lowpass_kernel_size=31,
    lowpass_sigma=5.0,
):
    raw_residual = (inp - target_t).detach()
    if mode == 'residual':
        return target_r, raw_residual
    if mode == 'lowpass_residual':
        return (
            gaussian_lowpass(raw_residual, lowpass_kernel_size, lowpass_sigma),
            raw_residual,
        )
    if mode == 'residual_lowpass_aux':
        return target_r, raw_residual
    raise ValueError(f'Unsupported reflection target mode: {mode}')


@MODEL_REGISTRY.register()
class RDNetModel(BaseModel):
    """
    This file defines the training process of RDNet and RDNet+.

    Please refer to the paper for more details:
        
        Reversible Decoupling Network for Single Image Reflection Removal (CVPR 2025).

        Reversible Adaptor for Single Image Reflection Removal (Preprint).
    
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_grad = None
        reflection_target = self.opt['train'].get('reflection_target', {})
        self.reflection_target_mode = reflection_target.get('mode', 'residual')
        self.reflection_lowpass_kernel_size = int(
            reflection_target.get('lowpass_kernel_size', 31)
        )
        self.reflection_lowpass_sigma = float(
            reflection_target.get('lowpass_sigma', 5.0)
        )
        self.reflection_lowpass_aux_weight = float(
            reflection_target.get('lowpass_aux_weight', 0.0)
        )
        self._validate_reflection_target_config()

    def _validate_reflection_target_config(self):
        if self.reflection_target_mode not in {
            'residual',
            'lowpass_residual',
            'residual_lowpass_aux',
        }:
            raise ValueError(
                'Unsupported reflection target mode: '
                f'{self.reflection_target_mode}. Use residual, lowpass_residual, '
                'or residual_lowpass_aux.'
            )
        if self.reflection_lowpass_kernel_size <= 0 or self.reflection_lowpass_kernel_size % 2 == 0:
            raise ValueError('reflection lowpass kernel size must be a positive odd integer')
        if self.reflection_lowpass_sigma <= 0:
            raise ValueError('reflection lowpass sigma must be positive')
        if self.reflection_lowpass_aux_weight < 0:
            raise ValueError('reflection lowpass auxiliary weight must be non-negative')

    def _gaussian_lowpass(self, img):
        return gaussian_lowpass(
            img,
            self.reflection_lowpass_kernel_size,
            self.reflection_lowpass_sigma,
        )

    def _build_reflection_target(self, inp, target_t, target_r):
        return build_reflection_target(
            inp,
            target_t,
            target_r,
            mode=self.reflection_target_mode,
            lowpass_kernel_size=self.reflection_lowpass_kernel_size,
            lowpass_sigma=self.reflection_lowpass_sigma,
        )

    @staticmethod
    def _match_target_size(target, output):
        if target.shape[-2:] == output.shape[-2:]:
            return target
        return F.interpolate(
            target,
            size=output.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])

        if not hasattr(self, 'cri_perceptual') or self.cri_perceptual is None:
            if self.opt['train'].get('perceptual_opt'):
                self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt'])

        if not hasattr(self, 'cri_grad') or self.cri_grad is None:
            if self.opt['train'].get('grad_opt'):
                self.cri_grad = build_loss(self.opt['train']['grad_opt'])


    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch (dict): Input batch containing 'input', 'target_t', 'target_r'.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Total loss.
        """
        # Get inputs
        inp = batch['input']
        target_t = batch['target_t']
        target_r = batch['target_r']
        reflection_target, raw_reflection_residual = self._build_reflection_target(
            inp, target_t, target_r
        )

        # Forward pass
        x_cls_out, x_img_out = self.net_g(inp)
        output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]

        # Calculate losses
        loss_dict = OrderedDict()
        pix_t_loss_list = []
        pix_r_loss_list = []
        per_loss_list = []
        grad_loss_list = []
        pix_r_raw_loss_list = []
        pix_r_low_aux_loss_list = []

        for i, out_imgs in enumerate(x_img_out):
            out_t, out_r = out_imgs[:, :3, ...], out_imgs[:, 3:, ...]
            target_t_i = self._match_target_size(target_t, out_t)
            reflection_target_i = self._match_target_size(reflection_target, out_r)
            # Pixel loss
            l_g_pix_t = self.cri_pix(out_t, target_t_i)
            pix_t_loss_list.append(l_g_pix_t)
            l_g_pix_r = self.cri_pix(out_r, reflection_target_i)
            pix_r_loss_list.append(l_g_pix_r)

            # Perceptual loss
            l_g_percep_t, _ = self.cri_perceptual(out_t, target_t_i)
            if l_g_percep_t is not None:
                per_loss_list.append(l_g_percep_t)

            # Gradient loss
            l_g_grad = self.cri_grad(out_t, target_t_i)
            grad_loss_list.append(l_g_grad)

            if self.reflection_target_mode == 'lowpass_residual':
                raw_reflection_residual_i = self._match_target_size(
                    raw_reflection_residual, out_r
                )
                l_g_pix_r_raw = self.cri_pix(
                    out_r.detach(), raw_reflection_residual_i
                )
                pix_r_raw_loss_list.append(l_g_pix_r_raw)
            elif self.reflection_target_mode == 'residual_lowpass_aux':
                raw_reflection_residual_i = self._match_target_size(
                    raw_reflection_residual, out_r
                )
                low_out_r = self._gaussian_lowpass(out_r)
                low_raw_residual = self._gaussian_lowpass(raw_reflection_residual_i)
                l_g_pix_r_low_aux = self.cri_pix(low_out_r, low_raw_residual)
                pix_r_low_aux_loss_list.append(l_g_pix_r_low_aux)

        # Apply weights to losses
        l_g_pix_t = self.calculate_weighted_loss(pix_t_loss_list)
        l_g_pix_r = self.calculate_weighted_loss(pix_r_loss_list)
        l_g_percep_t = self.calculate_weighted_loss(per_loss_list)
        l_g_grad = self.calculate_weighted_loss(grad_loss_list)
        l_g_pix_r_low_aux = None
        if pix_r_low_aux_loss_list:
            l_g_pix_r_low_aux = self.calculate_weighted_loss(pix_r_low_aux_loss_list)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_percep_t'] = l_g_percep_t
        loss_dict['l_g_grad'] = l_g_grad
        l_g_total = l_g_pix_t + l_g_pix_r + l_g_percep_t + l_g_grad
        if pix_r_low_aux_loss_list:
            weighted_low_aux = l_g_pix_r_low_aux * self.reflection_lowpass_aux_weight
            loss_dict['l_g_pix_r_low_aux'] = weighted_low_aux
            l_g_total = l_g_total + weighted_low_aux
        if pix_r_raw_loss_list:
            loss_dict['l_g_pix_r_raw_residual'] = self.calculate_weighted_loss(
                pix_r_raw_loss_list
            )

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=False)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_clean
        self.last_output_reflection = output_reflection
        self.last_target_t = target_t

        return l_g_total
    
    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            x_cls_out, x_img_out = model(inp)
            output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]
            self.output = [output_clean, output_reflection]

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            list: List of parameter groups.
        """
        train_opt = self.opt['train']

        # Setup different parameter groups with their learning rates
        params_lr = [
            {'params': self.net_g.get_baseball_params(), 'lr': train_opt['optim_g']['baseball_lr']},
            {'params': self.net_g.get_other_params(), 'lr': train_opt['optim_g']['other_lr']},
        ]

        # Get optimizer configuration without modifying original config
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items()
                        if k not in ['type', 'baseball_lr', 'other_lr']}

        return {
            'optim_type': optim_type,
            'params': params_lr,
            **optim_config,
        }

    def calculate_weighted_loss(self, loss_list):
        """Calculate weighted loss.
        This file gives a default implementation of calculating multi-scale weighted loss.
        Users can implement their own weighted loss function in the model file.
        
        Args:
            loss_list (list): List of losses at different scales.
        """
 
        weights = [i / len(loss_list) for i in range(1, len(loss_list) + 1)]
        
        while len(weights) < len(loss_list):
            weights.append(1.0)
        weights = weights[:len(loss_list)]
        return sum(w * loss for w, loss in zip(weights, loss_list))
