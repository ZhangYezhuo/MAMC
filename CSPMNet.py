"""Self-contained PyTorch implementation of CSPMNet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def morlet_wavelet_1d(length: int, center_freq: float, sigma: float) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) - length // 2
    gaussian = torch.exp(-(t**2) / (2.0 * sigma**2))
    carrier = torch.exp(1j * 2.0 * math.pi * center_freq * t)
    psi = gaussian.to(torch.complex64) * carrier.to(torch.complex64)
    return psi / psi.abs().sum()


def build_filter_bank(J: int, Q: int, filter_len: int) -> list[torch.Tensor]:
    wavelets = []
    for j in range(1, J * Q + 1):
        scale = 2.0 ** (j / Q)
        center_freq = 0.5 / scale
        sigma = scale
        wavelets.append(morlet_wavelet_1d(filter_len, center_freq, sigma))
    return wavelets


def shift_right_with_repeat(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    if steps >= x.shape[-1]:
        raise ValueError(f"steps must be smaller than signal length, got steps={steps}")
    prefix = x[..., :1].expand(*x.shape[:-1], steps)
    return torch.cat([prefix, x[..., :-steps]], dim=-1)


class ScaledAdditiveAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.query_projection = nn.Linear(feature_dim, attention_dim)
        self.key_projection = nn.Linear(feature_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.last_attention_weights = None

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if query.dim() != 2:
            raise ValueError(f"Expected query shape (B, C), got {tuple(query.shape)}")
        if values.dim() != 3:
            raise ValueError(f"Expected values shape (B, T, C), got {tuple(values.shape)}")
        if query.shape[0] != values.shape[0] or query.shape[1] != values.shape[2]:
            raise ValueError(
                "Query and values must share batch and feature dimensions, got "
                f"query={tuple(query.shape)}, values={tuple(values.shape)}"
            )

        projected_query = self.query_projection(query).unsqueeze(1)
        projected_keys = self.key_projection(values)
        scores = self.score_projection(torch.tanh(projected_query + projected_keys)).squeeze(-1)
        scores = scores / math.sqrt(self.attention_dim)

        attention_weights = F.softmax(scores, dim=1)
        self.last_attention_weights = attention_weights.detach()
        attention_weights = self.dropout(attention_weights)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        return self.layer_norm(context)


class TemporalAttentionFusionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        mixer_channels: int = 128,
        rnn_hidden: int = 96,
        feature_dim: int = 64,
        dense_hidden: int = 96,
    ) -> None:
        super().__init__()
        self.mixer_channels = mixer_channels
        self.rnn_hidden = rnn_hidden
        self.feature_dim = feature_dim
        self.rnn_feature_dim = 2 * rnn_hidden

        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels, mixer_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mixer_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.rnn = nn.GRU(
            input_size=mixer_channels,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_norm = nn.BatchNorm1d(self.rnn_feature_dim)
        self.rnn_dropout = nn.Dropout(0.3)
        self.attention = ScaledAdditiveAttention(
            feature_dim=self.rnn_feature_dim,
            attention_dim=rnn_hidden,
            dropout=0.1,
        )
        self.fc1 = nn.Linear(self.rnn_feature_dim, dense_hidden)
        self.fc_norm1 = nn.BatchNorm1d(dense_hidden)
        self.fc_dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dense_hidden, feature_dim)
        self.fc_norm2 = nn.BatchNorm1d(feature_dim)
        self.fc_dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _batch_norm_sequence(x: torch.Tensor, batch_norm: nn.BatchNorm1d) -> torch.Tensor:
        x = batch_norm(x.transpose(1, 2))
        return x.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mixer(x)
        x = x.transpose(1, 2).contiguous()
        x, _ = self.rnn(x)
        x = self._batch_norm_sequence(x, self.rnn_norm)
        x = self.rnn_dropout(x)

        query = x[:, -1, :]
        x = self.attention(query, x)

        x = F.relu(self.fc_norm1(self.fc1(x)))
        x = self.fc_dropout1(x)
        feature = F.relu(self.fc_norm2(self.fc2(x)))
        feature = self.fc_dropout2(feature)
        logits = self.fc_out(feature)
        return feature, logits


class FreeConvSubbandFrontEnd(nn.Module):
    def __init__(self, J: int = 2, Q: int = 4, filter_len: int = 33) -> None:
        super().__init__()
        self.J = int(J)
        self.Q = int(Q)
        self.filter_len = int(filter_len)
        wavelets = build_filter_bank(self.J, self.Q, self.filter_len)
        self.num_subbands = len(wavelets)
        weight_real = torch.stack([wavelet.real for wavelet in wavelets], dim=0).unsqueeze(1)
        weight_imag = torch.stack([wavelet.imag for wavelet in wavelets], dim=0).unsqueeze(1)
        self.weight_real = nn.Parameter(weight_real)
        self.weight_imag = nn.Parameter(weight_imag)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input of shape (B, 2, T), got {tuple(x.shape)}")
        pad = self.filter_len // 2
        x_real = x[:, 0:1, :]
        x_imag = x[:, 1:2, :]
        y_real = F.conv1d(x_real, self.weight_real, padding=pad) - F.conv1d(
            x_imag, self.weight_imag, padding=pad
        )
        y_imag = F.conv1d(x_real, self.weight_imag, padding=pad) + F.conv1d(
            x_imag, self.weight_real, padding=pad
        )
        signal_len = x.shape[-1]
        return y_real[..., :signal_len], y_imag[..., :signal_len]


class SubbandPhaseMotionFeatureMap(nn.Module):
    def __init__(self, lags: tuple[int, ...] = (1, 2, 4, 8), eps: float = 1e-6) -> None:
        super().__init__()
        self.lags = tuple(lags)
        self.eps = float(eps)
        if not self.lags:
            raise ValueError("SubbandPhaseMotionFeatureMap requires at least one lag")
        if any(lag <= 0 for lag in self.lags):
            raise ValueError(f"All lags must be positive, got lags={self.lags}")

    def output_channels(self, num_subbands: int) -> int:
        return 3 * int(num_subbands) * (1 + len(self.lags))

    def _base_features(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> list[torch.Tensor]:
        magnitude = torch.sqrt(z_real.pow(2) + z_imag.pow(2) + self.eps)
        return [torch.log1p(magnitude), z_real, z_imag]

    def _delta_features(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        lag: int,
    ) -> list[torch.Tensor]:
        prev_real = shift_right_with_repeat(z_real, lag)
        prev_imag = shift_right_with_repeat(z_imag, lag)
        delta_real = z_real * prev_real + z_imag * prev_imag
        delta_imag = z_imag * prev_real - z_real * prev_imag
        delta_mag = torch.sqrt(delta_real.pow(2) + delta_imag.pow(2) + self.eps)
        return [torch.log1p(delta_mag), delta_real, delta_imag]

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> torch.Tensor:
        if z_real.shape != z_imag.shape:
            raise ValueError(
                "Subband real and imaginary responses must share shape, got "
                f"real={tuple(z_real.shape)}, imag={tuple(z_imag.shape)}"
            )
        if z_real.dim() != 3:
            raise ValueError(f"Expected subband responses of shape (B, S, T), got {tuple(z_real.shape)}")
        if max(self.lags) >= z_real.shape[-1]:
            raise ValueError(
                f"Maximum lag must be smaller than response length, got max_lag={max(self.lags)} "
                f"and length={z_real.shape[-1]}"
            )

        features = self._base_features(z_real, z_imag)
        for lag in self.lags:
            features.extend(self._delta_features(z_real, z_imag, lag))
        return torch.cat(features, dim=1)


class SubbandPhaseMotionAttnRNNBase(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim: int,
        front_end: nn.Module,
        lags: tuple[int, ...] = (1, 2, 4, 8),
        mixer_channels: int = 128,
        rnn_hidden: int = 96,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if len(input_dim) < 2 or input_dim[0] != 2:
            raise ValueError(
                "SubbandPhaseMotionAttnRNNBase expects 2-channel I/Q input, "
                f"got input_dim={input_dim}"
            )

        self.front_end = front_end
        self.feature_map = SubbandPhaseMotionFeatureMap(lags=lags)
        self.feature_channels = self.feature_map.output_channels(front_end.num_subbands)
        self.feature_norm = nn.BatchNorm1d(self.feature_channels)
        self.fusion = TemporalAttentionFusionHead(
            in_channels=self.feature_channels,
            num_classes=output_dim,
            mixer_channels=mixer_channels,
            rnn_hidden=rnn_hidden,
            feature_dim=feature_dim,
        )

    def extract_subband_phase_motion_map(self, x: torch.Tensor) -> torch.Tensor:
        z_real, z_imag = self.front_end(x)
        features = self.feature_map(z_real, z_imag)
        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {features.shape[1]}"
            )
        return self.feature_norm(features)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        features = self.extract_subband_phase_motion_map(x)
        feature, logits = self.fusion(features)
        if return_feature:
            return feature, logits
        return None, logits

    def count_parameters(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        front_end_trainable = sum(
            parameter.numel() for parameter in self.front_end.parameters() if parameter.requires_grad
        )
        return {
            "total_trainable": trainable,
            "total_all": total,
            "front_end_trainable": front_end_trainable,
            "front_end_type": self.front_end.__class__.__name__,
            "num_subbands": self.front_end.num_subbands,
            "lags": self.feature_map.lags,
            "feature_channels": self.feature_channels,
            "mixer_channels": self.fusion.mixer_channels,
            "rnn_hidden": self.fusion.rnn_hidden,
            "feature_dim": self.fusion.feature_dim,
        }


class FreeConvSubbandPhaseMotionAttnRNN(SubbandPhaseMotionAttnRNNBase):
    def __init__(
        self,
        input_dim,
        output_dim: int,
        J: int = 2,
        Q: int = 4,
        filter_len: int = 33,
        lags: tuple[int, ...] = (1, 2, 4, 8),
        mixer_channels: int = 128,
        rnn_hidden: int = 96,
        feature_dim: int = 64,
    ) -> None:
        front_end = FreeConvSubbandFrontEnd(
            J=J,
            Q=Q,
            filter_len=filter_len,
        )
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            front_end=front_end,
            lags=lags,
            mixer_channels=mixer_channels,
            rnn_hidden=rnn_hidden,
            feature_dim=feature_dim,
        )


class CSPMNet(FreeConvSubbandPhaseMotionAttnRNN):
    """CSPMNet model for 2-channel I/Q automatic modulation classification."""


__all__ = [
    "CSPMNet",
    "FreeConvSubbandPhaseMotionAttnRNN",
    "FreeConvSubbandFrontEnd",
    "SubbandPhaseMotionFeatureMap",
]
