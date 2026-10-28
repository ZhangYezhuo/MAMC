"""Adapters for controlled comparisons; preserve original feature/model logic."""
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

from torch import nn


@dataclass(frozen=True)
class BaselineConfig:
    architecture: str
    num_classes: int
    length: int


class Baseline(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        if config.architecture == "legacy":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
            from raw_model.get_model import get_model
            self.model = get_model(SimpleNamespace(arch="MAMCA", class_names=range(config.num_classes),
                                                   length_train=config.length), device)
        elif config.architecture == "cspmnet":
            from CSPMNet import CSPMNet
            self.model = CSPMNet((2, config.length), config.num_classes)
        else:
            raise ValueError("Unknown baseline")

    def forward(self, x):
        y = self.model(x)
        return y[-1] if isinstance(y, tuple) else y

    def optimizer_groups(self, weight_decay):
        groups = [[], []]
        for p in self.parameters():
            groups[int(p.ndim < 2 or getattr(p, "_no_weight_decay", False))].append(p)
        return [{"params": groups[0], "weight_decay": weight_decay},
                {"params": groups[1], "weight_decay": 0.0}]
