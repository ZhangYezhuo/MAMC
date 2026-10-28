"""Real I/Q data loading. No upsampling or repetition to fake longer signals."""
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    def __init__(self, path, length=None):
        path = Path(path)
        if path.suffix in (".pkl", ".pickle"):
            # RadioML2016 dictionaries: (modulation, snr) -> (N,2,L).
            with path.open("rb") as f:
                data = pickle.load(f, encoding="latin1")
            self.classes = sorted({str(k[0]) for k in data})
            xs, ys, snrs = [], [], []
            for (mod, snr), values in sorted(data.items()):
                xs.append(np.asarray(values))
                ys.append(np.full(len(values), self.classes.index(str(mod))))
                snrs.append(np.full(len(values), float(snr)))
            x, y, snr = np.concatenate(xs), np.concatenate(ys), np.concatenate(snrs)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                x, y = data["X"], data["Y"]
                snr = data["SNR"].reshape(-1) if "SNR" in data else np.zeros(len(x))
                self.classes = data["classes"].astype(str).tolist() if "classes" in data else None
        elif path.suffix in (".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as data:
                x, y = data["X"][:], data["Y"][:]
                key = "SNR" if "SNR" in data else "Z"
                snr = data[key][:].reshape(-1) if key in data else np.zeros(len(x))
            self.classes = None
        else:
            raise ValueError("Use RadioML .pkl, .npz (X,Y,SNR), or .h5/.hdf5 (X,Y,Z/SNR)")
        if x.ndim != 3:
            raise ValueError("X must be (N,2,L) or (N,L,2)")
        if x.shape[1] != 2 and x.shape[-1] == 2:
            x = x.transpose(0, 2, 1)
        if x.shape[1] != 2 or np.iscomplexobj(x):
            raise ValueError("Expected two real I/Q channels")
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(1)
        y = np.asarray(y).reshape(-1)
        if not np.isfinite(y).all() or np.any(y != np.floor(y)):
            raise ValueError("Y must contain integer class indices or one-hot labels")
        y = y.astype(np.int64)
        labels = np.unique(y)
        if not np.array_equal(labels, np.arange(len(labels))):
            raise ValueError("Class IDs must be contiguous from zero")
        self.classes = self.classes or [str(i) for i in labels]
        if len(self.classes) != len(labels) or len(y) != len(x) or len(snr) != len(x):
            raise ValueError("Inconsistent X/Y/SNR/classes dimensions")
        if length is not None:
            if length < 1 or length > x.shape[-1]:
                raise ValueError(f"Requested length {length}; actual samples have length {x.shape[-1]}. "
                                 "Provide genuinely longer observations; do not tile short signals.")
            x = x[..., :length]
        if not np.isfinite(x).all() or not np.isfinite(snr).all():
            raise ValueError("Nonfinite signal or SNR")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(y)
        self.snr = np.asarray(snr, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.snr[index]


def stratified_split(dataset, seed=42, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Need three positive train/validation/test fractions summing to one")
    rng = np.random.default_rng(seed)
    groups = {}
    for i, (label, snr) in enumerate(zip(dataset.y.tolist(), dataset.snr.tolist())):
        groups.setdefault((label, snr), []).append(i)
    splits = [[], [], []]
    for key, indices in groups.items():
        if len(indices) < 3:
            raise ValueError(f"Need >= 3 samples per class/SNR group, got {key}: {len(indices)}")
        indices = rng.permutation(indices)
        n_train = min(max(1, int(len(indices) * fractions[0])), len(indices) - 2)
        n_val = min(max(1, int(len(indices) * fractions[1])), len(indices) - n_train - 1)
        for split, part in zip(splits, np.split(indices, [n_train, n_train + n_val])):
            split.extend(part.tolist())
    return [np.array(sorted(split), dtype=np.int64) for split in splits]
