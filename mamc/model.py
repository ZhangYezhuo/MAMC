"""CSPM local feature extraction -> residual Mamba2 -> attention/mean pooling."""
from dataclasses import dataclass

import torch
from torch import nn

from .csp import CSPMFrontEnd
from .mamba2 import build_mamba2


@dataclass(frozen=True)
class MAMC2Config:
    num_classes: int = 11
    d_model: int = 32
    n_layer: int = 2
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    chunk_size: int = 64
    subbands: int = 4
    filter_len: int = 17
    lags: tuple[int, ...] = (1, 4)
    stride: int = 1
    dropout: float = 0.1
    backend: str = "official"
    fused: bool = True
    frontend: str = "csp"

    def __post_init__(self):
        for name in ("num_classes", "d_model", "n_layer", "d_state", "d_conv", "expand", "headdim", "chunk_size", "stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model * self.expand % self.headdim:
            raise ValueError("d_model * expand must be divisible by headdim")
        if self.chunk_size & (self.chunk_size - 1):
            raise ValueError("chunk_size must be a power of two")
        if self.backend not in ("official", "torch") or self.frontend not in ("csp", "raw"):
            raise ValueError("backend must be official/torch and frontend must be csp/raw")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class ResidualMamba2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = build_mamba2(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


class MAMC2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MAMC2Config()
        c = self.config
        if c.frontend == "csp":
            self.frontend = CSPMFrontEnd(c.d_model, c.subbands, c.filter_len, c.lags, c.stride)
        else:
            self.frontend = nn.Sequential(nn.Conv1d(2, c.d_model, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualMamba2Block(c) for _ in range(c.n_layer)))
        self.norm = nn.LayerNorm(c.d_model)
        # One score per token, no pairwise attention matrix; O(L).
        self.attention = nn.Linear(c.d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * c.d_model), nn.Dropout(c.dropout),
                                  nn.Linear(2 * c.d_model, c.num_classes))

    def forward(self, x, return_feature=False):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] < 1:
            raise ValueError("Expected real I/Q tensor (B, 2, L)")
        if self.config.backend == "official":
            if x.device.type != "cuda":
                raise RuntimeError("Official Mamba2 needs CUDA; use backend='torch' on CPU")
        x = self.frontend(x)
        if self.config.frontend == "raw":
            if self.config.stride > 1:
                x = torch.nn.functional.avg_pool1d(x, self.config.stride, self.config.stride, ceil_mode=True)
            x = x.transpose(1, 2).contiguous()
        x = self.norm(self.blocks(x))
        weights = torch.softmax(self.attention(x).float(), dim=1).to(x.dtype)
        feature = torch.cat((x.mean(1), (x * weights).sum(1)), dim=-1)
        logits = self.head(feature)
        return (feature, logits) if return_feature else logits

    def optimizer_groups(self, weight_decay):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
