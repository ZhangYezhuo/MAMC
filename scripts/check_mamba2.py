#!/usr/bin/env python
"""Distinguish package import, official CUDA execution, and portable execution."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mamc import MAMC2, MAMC2Config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-official", action="store_true", help="Require a passing official fused CUDA path")
    p.add_argument("--output", default="outputs/mamba2-environment.json")
    a = p.parse_args()
    report = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "packages": {}}
    for name in ["mamba-ssm", "causal-conv1d", "triton", "einops", "transformers"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    for fused, amp in [(True, False), (False, False), (True, True), (False, True)]:
        key = ("official_fused" if fused else "official_unfused") + ("_amp" if amp else "")
        try:
            from mamba_ssm import Mamba2
            m = Mamba2(d_model=32, d_state=32, headdim=32, chunk_size=64,
                       use_mem_eff_path=fused).to(device)
            x = torch.randn(2, 128, 32, device=device, requires_grad=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp and device == "cuda"):
                y = m(x)
                loss = y.float().square().mean()
            loss.backward()
            assert torch.isfinite(y).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            if device == "cuda":
                torch.cuda.synchronize()
            report[key] = {"success": True, "finite": bool(torch.isfinite(y).all())}
        except Exception as exc:
            report[key] = {"success": False, "error_type": type(exc).__name__, "error": str(exc),
                           "traceback": traceback.format_exc()}
    try:
        model = MAMC2(MAMC2Config(backend="torch")).to(device)
        logits = model(torch.randn(2, 2, 129, device=device))
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        report["portable"] = {"success": True, "shape": list(logits.shape), "config": asdict(model.config)}
    except Exception as exc:
        report["portable"] = {"success": False, "error": repr(exc)}
    if device == "cuda":
        try:
            model = MAMC2(MAMC2Config(backend="torch")).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(torch.randn(2, 2, 129, device=device))
                loss = logits.float().square().mean()
            loss.backward()
            assert torch.isfinite(logits).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            report["portable_amp"] = {"success": True, "logits_dtype": str(logits.dtype)}
        except Exception as exc:
            report["portable_amp"] = {"success": False, "error_type": type(exc).__name__, "error": str(exc)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("official")}, indent=2))
    for k in ("official_fused", "official_unfused", "official_fused_amp", "official_unfused_amp"):
        print(k, {n: v for n, v in report[k].items() if n != "traceback"})
    official_ok = any(report[key]["success"] for key in ("official_fused", "official_fused_amp"))
    return 0 if report["portable"]["success"] and (official_ok or not a.require_official) else 1


if __name__ == "__main__":
    raise SystemExit(main())
