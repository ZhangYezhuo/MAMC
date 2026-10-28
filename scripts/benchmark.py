#!/usr/bin/env python
"""Random I/Q latency and memory, with CUDA synchronization and saved provenance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mamc import MAMC2, MAMC2Config


def create_model(a, length):
    if a.model == "cspmnet":
        from CSPMNet import CSPMNet
        return CSPMNet((2, length), 11)
    if a.model == "legacy":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
        from raw_model.get_model import get_model
        from types import SimpleNamespace
        return get_model(SimpleNamespace(arch="MAMCA", class_names=list(range(11)), length_train=length), a.device)
    return MAMC2(MAMC2Config(backend=a.backend, stride=a.stride, frontend=a.frontend, d_model=a.width, n_layer=a.layers))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["mamc2", "legacy", "cspmnet"], default="mamc2")
    p.add_argument("--backend", choices=["official", "torch"], default="official")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frontend", choices=["csp", "raw"], default="csp")
    p.add_argument("--training", action="store_true", help="Forward + loss + backward, no optimizer step")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output", default="outputs/benchmark.json")
    a = p.parse_args()
    if min(a.lengths) < 1 or a.repeats < 1 or a.warmup < 0 or a.batch_size < 1:
        p.error("Invalid benchmark sizes")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cuda = a.device.startswith("cuda")
    device = torch.device(a.device)
    rows = []
    report = {"args": vars(a), "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device) if cuda else None, "results": rows,
              "input": "random normal; no accuracy measured", "seed": 42,
              "timing": "synchronized wall clock; includes Python overhead; excludes optimizer and transfer"}
    for length in a.lengths:
        model, x = None, None
        try:
            model = create_model(a, length).to(device).train(a.training)
            x = torch.randn(a.batch_size, 2, length, device=device)
            def step():
                if a.training:
                    model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(a.training), torch.autocast(device_type=device.type,
                        dtype=torch.float16, enabled=a.amp and cuda):
                    y = model(x)
                    if isinstance(y, tuple):
                        y = y[-1]
                    if a.training:
                        loss = y.float().square().mean()
                if a.training:
                    loss.backward()
            for _ in range(a.warmup):
                step()
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
            else:
                baseline = 0
            samples = []
            for _ in range(a.repeats):
                if cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                step()
                if cuda:
                    torch.cuda.synchronize(device)
                samples.append(1000 * (time.perf_counter() - start))
            peak = torch.cuda.max_memory_allocated(device) if cuda else None
            row = {"length": length, "parameters": sum(p.numel() for p in model.parameters()),
                   "median_ms": statistics.median(samples), "min_ms": min(samples),
                   "p95_ms": sorted(samples)[min(len(samples)-1, int(len(samples)*.95))],
                   "signals_per_second": 1000 * a.batch_size / statistics.median(samples),
                   "baseline_allocated_mib": baseline / 2**20 if cuda else None,
                   "peak_allocated_mib": peak / 2**20 if cuda else None,
                   "incremental_peak_mib": (peak - baseline) / 2**20 if cuda else None}
            if a.model == "mamc2":
                report["config"] = asdict(model.config)
        except Exception as exc:
            row = {"length": length, "error_type": type(exc).__name__, "error": str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, x
        if cuda:
            torch.cuda.empty_cache()
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(report, indent=2))
    if any("error" in row for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
