"""Mamba2 adapter and portable SSD implementation.

Equations and parameter layout follow state-spaces/mamba's Mamba2 and
ssd_minimal.py (Tri Dao, Albert Gu, Apache-2.0). This is an independently
written, restricted implementation: full SSM width, one B/C group, scalar D
per head, gated RMSNorm; no incremental cache or tensor parallelism.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def affine_prefix_states(decays, drives):
    """Exclusive affine prefix with a work-efficient binary tree.

    Compose (a,b): state -> a*state+b. Upsweep + downsweep use O(chunks)
    total state work and O(log chunks) sequential launches. Decays are never
    inverted, avoiding unstable division by underflowed cumulative products.
    """
    count = decays.shape[1]
    tree_size = 1 << (count - 1).bit_length()
    if tree_size != count:
        decays = torch.cat((decays, decays.new_ones(decays.shape[0], tree_size-count, decays.shape[2])), 1)
        drives = torch.cat((drives, drives.new_zeros(drives.shape[0], tree_size-count, *drives.shape[2:])), 1)
    levels = [(decays, drives)]
    while decays.shape[1] > 1:
        right_decay = decays[:, 1::2]
        drives = right_decay[..., None, None] * drives[:, ::2] + drives[:, 1::2]
        decays = right_decay * decays[:, ::2]
        levels.append((decays, drives))
    prefixes = torch.zeros_like(drives)
    for level_decay, level_drive in reversed(levels[:-1]):
        right = level_decay[:, ::2, :, None, None] * prefixes + level_drive[:, ::2]
        prefixes = torch.stack((prefixes, right), dim=2).flatten(1, 2)
    return prefixes[:, :count]


def ssd_scan(x, dt, A, B, C, D, chunk_size):
    """Chunked SSD, O(L) for fixed chunk size; fp32 states and no L x L tensor.

    x: B,L,H,P; dt: B,L,H; B/C: B,L,N (one shared group).
    """
    x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
    batch, length, heads, head_dim = x.shape
    chunks = math.ceil(length / chunk_size)
    padding = chunks * chunk_size - length
    def partition(tensor):
        pad = (0, 0) * (tensor.ndim - 2) + (0, padding)
        tensor = F.pad(tensor, pad)
        return tensor.reshape(batch, chunks, chunk_size, *tensor.shape[2:])
    xs, ds, bs, cs = map(partition, (x, dt, B, C))
    increments = (ds * A).permute(0, 3, 1, 2)  # B,H,chunks,K
    cumulative = increments.cumsum(-1)
    # Independent within-chunk operations are batched, not launched per chunk.
    lower = torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool).tril(-1)
    segments = increments.unsqueeze(-1).expand(-1, -1, -1, -1, chunk_size)
    segments = segments.masked_fill(~lower, 0).cumsum(-2)
    causal = torch.ones_like(lower).tril()
    decay = segments.masked_fill(~causal, -torch.inf).exp()
    cb = torch.einsum("bctn,bcsn->bcts", cs, bs)
    xd = xs * ds.unsqueeze(-1)
    within = torch.einsum("bhcts,bcshp->bcthp", decay * cb.unsqueeze(1), xd)
    tail_decay = (cumulative[..., -1:] - cumulative).exp().permute(0, 2, 3, 1)
    # Contract T directly; a three-input einsum can materialize L*H*P*N
    # when opt_einsum is absent, wasting memory on long observations.
    drives = torch.einsum("bcthp,bctn->bchpn", xd * tail_decay.unsqueeze(-1), bs)
    chunk_decay = cumulative[..., -1].exp().permute(0, 2, 1)
    incoming_states = affine_prefix_states(chunk_decay, drives)
    incoming = torch.einsum("bctn,bchpn->bcthp", cs, incoming_states)
    incoming = incoming * cumulative.exp().permute(0, 2, 3, 1).unsqueeze(-1)
    y = within + incoming + xs * D[None, None, None, :, None]
    return y.reshape(batch, chunks * chunk_size, heads, head_dim)[:, :length]



class GatedRMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x, z):
        y = x.float() * F.silu(z.float())
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


class TorchMamba2(nn.Module):
    """Portable Mamba2 with upstream-compatible parameter names for this subset.

    This backend supports CPU/Turing and debugging; it is not the Triton fast path.
    Select it explicitly. Never silently substitute it for the official backend.
    """
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, headdim=32, chunk_size=64):
        super().__init__()
        self.d_inner, self.d_state = d_model * expand, d_state
        self.headdim, self.chunk_size, self.d_conv = headdim, chunk_size, d_conv
        self.nheads = self.d_inner // headdim
        conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.empty(self.nheads).uniform_(math.log(.001), math.log(.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        for parameter in (self.dt_bias, self.A_log, self.D):
            parameter._no_weight_decay = True
        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :u.shape[1]].transpose(1, 2))
        x, B, C = xbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(*x.shape[:2], self.nheads, self.headdim)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        y = ssd_scan(x, dt, -self.A_log.float().exp(), B, C, self.D, self.chunk_size)
        y = y.flatten(2).to(u.dtype)
        return self.out_proj(self.norm(y, z))


def build_mamba2(config):
    kwargs = dict(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                  expand=config.expand, headdim=config.headdim, chunk_size=config.chunk_size)
    if config.backend == "torch":
        return TorchMamba2(**kwargs)
    try:
        from mamba_ssm import Mamba2
    except (ImportError, OSError) as exc:
        raise RuntimeError("Official Mamba2 import failed. See the installation section in README.md, "
                           "or explicitly select backend='torch' for portable SSD.") from exc
    return Mamba2(**kwargs, use_mem_eff_path=config.fused)
