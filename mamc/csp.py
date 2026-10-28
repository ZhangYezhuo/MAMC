"""Lightweight adaptation of the supplied ../CSPMNet.py front end.

Retains trainable Morlet-initialized complex subbands and lagged conjugate
products. The BiGRU is replaced by Mamba2 in model.py.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CSPMFrontEnd(nn.Module):
    def __init__(self, width=32, subbands=4, filter_len=17, lags=(1, 4), stride=1):
        super().__init__()
        if subbands < 1 or filter_len < 3 or filter_len % 2 != 1:
            raise ValueError("subbands must be positive and filter_len must be odd and >= 3")
        if not lags or any(lag < 1 for lag in lags) or stride < 1:
            raise ValueError("positive lags and stride are required")
        self.lags, self.stride = tuple(lags), stride
        t = torch.arange(filter_len, dtype=torch.float32) - filter_len // 2
        scales = 2 ** (torch.arange(1, subbands + 1, dtype=torch.float32) / (subbands / 2))
        wavelets = torch.exp(-t.square() / (2 * scales[:, None].square())) * torch.exp(
            1j * 2 * math.pi * (0.5 / scales[:, None]) * t)
        wavelets = wavelets / wavelets.abs().sum(-1, keepdim=True)
        self.weight_real = nn.Parameter(wavelets.real[:, None].contiguous())
        self.weight_imag = nn.Parameter(wavelets.imag[:, None].contiguous())
        channels = 3 * subbands * (1 + len(lags))
        self.norm = nn.BatchNorm1d(channels)
        self.project = nn.Conv1d(channels, width, 1, bias=False)
        self.local = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, groups=width, bias=False),
            nn.BatchNorm1d(width), nn.SiLU(), nn.Conv1d(width, width, 1, bias=False))
        self.raw_skip = nn.Conv1d(2, width, 1, bias=False)

    @staticmethod
    def features(real, imag):
        # Compute products in float32 under AMP to avoid fp16 overflow.
        real, imag = real.float(), imag.float()
        return [torch.log1p(torch.sqrt(real.square() + imag.square() + 1e-6)), real, imag]

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] <= max(self.lags):
            raise ValueError(f"Expected (B, 2, L) with L > {max(self.lags)}, got {tuple(x.shape)}")
        # One real convolution implements the same complex convolution as CSPMNet.
        kernel = torch.cat((torch.cat((self.weight_real, -self.weight_imag), 1),
                            torch.cat((self.weight_imag, self.weight_real), 1)), 0)
        real, imag = F.conv1d(x, kernel, padding=kernel.shape[-1] // 2).chunk(2, dim=1)
        real, imag = real.float(), imag.float()
        features = self.features(real, imag)
        for lag in self.lags:
            prev_r = F.pad(real[..., :-lag], (lag, 0), mode="replicate")
            prev_i = F.pad(imag[..., :-lag], (lag, 0), mode="replicate")
            features.extend(self.features(real * prev_r + imag * prev_i,
                                          imag * prev_r - real * prev_i))
        features = self.norm(torch.cat(features, 1).to(x.dtype))
        # Extract phase motion at the original sampling rate, then reduce the
        # local mixer and SSM token counts together. Default keeps all samples.
        raw = x
        if self.stride > 1:
            features = F.avg_pool1d(features, self.stride, self.stride, ceil_mode=True)
            raw = F.avg_pool1d(raw, self.stride, self.stride, ceil_mode=True)
        features = self.project(features)
        features = features + self.local(features) + self.raw_skip(raw)
        return features.transpose(1, 2).contiguous()
