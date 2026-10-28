#!/usr/bin/env python
"""Rebuild matching Mamba2-era CUDA extensions against the active PyTorch.

Use when release wheels import with undefined C++ symbols. Only setup.py's
architecture list and NVCC job count are changed; model/kernel sources are intact.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-home", default="/usr/local/cuda-12.4")
    p.add_argument("--work-dir", default=str(Path.home() / ".cache/mamc/build"))
    p.add_argument("--packages", nargs="+", choices=["mamba-ssm", "causal-conv1d"],
                   default=["mamba-ssm", "causal-conv1d"])
    a = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        p.error("A visible NVIDIA GPU is needed to select a build architecture")
    cuda = Path(a.cuda_home).resolve()
    if not (cuda / "bin/nvcc").is_file():
        p.error("--cuda-home must be a single toolkit directory containing bin/nvcc")
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    versions = {"mamba-ssm": "2.2.6.post3", "causal-conv1d": "1.5.2"}
    provenance = {"torch": torch.__version__, "cuda_home": str(cuda), "arch": arch,
                  "abi": torch._C._GLIBCXX_USE_CXX11_ABI, "packages": {}}
    for name in a.packages:
        version = versions[name]
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
            metadata = json.load(response)
        source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
        with urllib.request.urlopen(source["url"]) as response:
            archive = response.read()
        checksum = hashlib.sha256(archive).hexdigest()
        if checksum != source["digests"]["sha256"]:
            raise RuntimeError("Source archive SHA256 mismatch")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            root = tar.getnames()[0].split("/")[0]
            tar.extractall(work, filter="data")
        package_dir = work / root
        setup = package_dir / "setup.py"
        original = setup.read_text()
        marker = "    # HACK: The compiler flag"
        if original.count(marker) != 1:
            raise RuntimeError("Unexpected upstream setup.py; inspect architecture selection manually")
        patched = original.replace(marker, f'    cc_flag = ["-gencode", "arch=compute_{arch},code=sm_{arch}"]\n\n' + marker)
        patched = patched.replace('return nvcc_extra_args + ["--threads", "4"]',
                                  'return nvcc_extra_args + ["--threads", "1"]')
        setup.write_text(patched)
        env = {**os.environ, "CUDA_HOME": str(cuda), "MAX_JOBS": "1",
               "MAMBA_FORCE_BUILD": "TRUE", "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"}
        subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                        "--no-build-isolation", str(package_dir)], env=env, check=True)
        provenance["packages"][name] = {"version": version, "url": source["url"], "sha256": checksum,
                                          "changes": "setup.py only: GPU architecture + NVCC threads=1"}
        (work / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("Build provenance:", work / "provenance.json")


if __name__ == "__main__":
    main()
