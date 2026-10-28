#!/usr/bin/env python
"""Reproducible AMC training without tllib; three disjoint class/SNR splits."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from mamc import MAMC2, MAMC2Config
from mamc.data import IQDataset, stratified_split
from mamc.baselines import Baseline, BaselineConfig


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    by_snr = {}
    confusion = torch.zeros(model.config.num_classes, model.config.num_classes, dtype=torch.long)
    for x, y, snr in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        pred = logits.argmax(-1)
        loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        hits = pred.eq(y).cpu()
        correct += hits.sum().item()
        total += len(y)
        for truth, prediction in zip(y.cpu().tolist(), pred.cpu().tolist()):
            confusion[truth, prediction] += 1
        for s, hit in zip(snr.tolist(), hits.tolist()):
            entry = by_snr.setdefault(str(s), [0, 0])
            entry[0] += int(hit)
            entry[1] += 1
    if not total:
        raise ValueError("Empty evaluation partition")
    return {"loss": loss / total, "accuracy": correct / total,
            "accuracy_by_snr": {s: c / n for s, (c, n) in by_snr.items()},
            "confusion_matrix": confusion.tolist(), "samples": total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="outputs/mamc2")
    p.add_argument("--model", choices=["mamc2", "legacy", "cspmnet"], default="mamc2")
    p.add_argument("--backend", choices=["official", "torch"], default="official")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frontend", choices=["csp", "raw"], default="csp")
    p.add_argument("--amp", action="store_true", help="CUDA fp16 AMP; fp32 parameters and SSD states")
    p.add_argument("--unfused", action="store_true")
    p.add_argument("--checkpoint", help="Evaluate checkpoint on its saved held-out split; no training")
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if not a.checkpoint and (output / "history.jsonl").exists():
        raise FileExistsError("Training output already has a history; choose a fresh --output directory")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False) if a.checkpoint else None
    length = checkpoint["length"] if checkpoint else a.length
    data = IQDataset(a.data, length)
    digest = hashlib.sha256()
    with Path(a.data).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    data_sha256 = digest.hexdigest()
    if checkpoint and checkpoint.get("data_sha256", data_sha256) != data_sha256:
        raise ValueError("Dataset SHA256 differs from the checkpoint's training dataset")
    if checkpoint:
        if "architecture" in checkpoint["config"]:
            config = BaselineConfig(**checkpoint["config"])
        else:
            config = MAMC2Config(**{**checkpoint["config"], "backend": a.backend, "fused": not a.unfused})
        if data.classes != checkpoint["classes"] or len(data) != checkpoint["dataset_size"]:
            raise ValueError("Checkpoint evaluation requires the original dataset and class order")
        splits = checkpoint["splits"]
    else:
        config = MAMC2Config(num_classes=len(data.classes), d_model=a.width, n_layer=a.layers,
                             stride=a.stride, backend=a.backend, frontend=a.frontend, fused=not a.unfused)
        if a.model != "mamc2":
            config = BaselineConfig(a.model, len(data.classes), data.x.shape[-1])
        splits = stratified_split(data, a.seed)
    loaders = [DataLoader(Subset(data, indices), batch_size=a.batch_size,
                          shuffle=(i == 0), num_workers=a.workers, pin_memory=a.device.startswith("cuda"))
               for i, indices in enumerate(splits)]
    amp = a.amp and a.device.startswith("cuda")
    model = (Baseline(config, a.device) if isinstance(config, BaselineConfig) else MAMC2(config)).to(a.device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, loaders[2], a.device, amp)
        (output / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    np.savez(output / "splits.npz", train=splits[0], validation=splits[1], test=splits[2])
    (output / "config.json").write_text(json.dumps({"model": asdict(config), "args": vars(a),
                                                   "classes": data.classes, "data_sha256": data_sha256}, indent=2))
    optimizer = torch.optim.AdamW(model.optimizer_groups(a.weight_decay), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -1
    for epoch in range(a.epochs):
        model.train()
        train_loss, samples = 0.0, 0
        for x, y, _ in loaders[0]:
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(a.device).type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)
            samples += len(y)
        scheduler.step()
        validation = evaluate(model, loaders[1], a.device, amp)
        row = {"epoch": epoch + 1, "train_loss": train_loss / samples, "validation": validation}
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if validation["accuracy"] > best:
            best = validation["accuracy"]
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch + 1,
                        "classes": data.classes, "length": data.x.shape[-1], "dataset_size": len(data),
                        "splits": [i.tolist() for i in splits], "seed": a.seed, "amp": amp, "data_sha256": data_sha256,
                        "validation": validation}, output / "best.pt")
    best_checkpoint = torch.load(output / "best.pt", map_location=a.device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test = evaluate(model, loaders[2], a.device, amp)
    (output / "test.json").write_text(json.dumps(test, indent=2))
    print("Held-out test:", json.dumps(test))


if __name__ == "__main__":
    main()
