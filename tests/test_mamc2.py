from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamc import MAMC2, MAMC2Config
from mamc.csp import CSPMFrontEnd
from mamc.mamba2 import TorchMamba2, ssd_scan
from mamc.data import IQDataset, stratified_split

torch.set_num_threads(2)


def recurrence(x, dt, A, B, C, D):
    state = x.new_zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1])
    ys = []
    for t in range(x.shape[1]):
        state = state * torch.exp(dt[:, t] * A)[:, :, None, None] + torch.einsum(
            "bhp,bn,bh->bhpn", x[:, t], B[:, t], dt[:, t])
        ys.append(torch.einsum("bhpn,bn->bhp", state, C[:, t]) + x[:, t] * D[None, :, None])
    return torch.stack(ys, 1)


@pytest.mark.parametrize("length", [1, 17, 64, 137])
def test_ssd_output_and_gradient_against_recurrence(length):
    torch.manual_seed(7)
    values = [torch.randn(2, length, 2, 4), torch.rand(2, length, 2) * .2,
              -torch.rand(2), torch.randn(2, length, 3), torch.randn(2, length, 3), torch.randn(2)]
    actual_inputs = [t.requires_grad_() for t in values]
    reference_inputs = [t.detach().clone().requires_grad_() for t in values]
    actual = ssd_scan(*actual_inputs, chunk_size=16)
    expected = recurrence(*reference_inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    weight = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weight).sum(), actual_inputs)
    expected_grads = torch.autograd.grad((expected * weight).sum(), reference_inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


def test_csp_matches_supplied_complex_frontend():
    from CSPMNet import FreeConvSubbandFrontEnd, SubbandPhaseMotionFeatureMap
    original = FreeConvSubbandFrontEnd(J=2, Q=2, filter_len=17)
    adapted = CSPMFrontEnd(subbands=4, filter_len=17)
    torch.testing.assert_close(original.weight_real, adapted.weight_real)
    torch.testing.assert_close(original.weight_imag, adapted.weight_imag)
    x = torch.randn(2, 2, 71)
    kernel = torch.cat((torch.cat((adapted.weight_real, -adapted.weight_imag), 1),
                        torch.cat((adapted.weight_imag, adapted.weight_real), 1)), 0)
    r, i = F.conv1d(x, kernel, padding=8).chunk(2, 1)
    ro, io = original(x)
    torch.testing.assert_close(r, ro)
    torch.testing.assert_close(i, io)
    expected = SubbandPhaseMotionFeatureMap(lags=(1, 4))(ro, io)
    features = adapted.features(r, i)
    for lag in adapted.lags:
        pr = F.pad(r[..., :-lag], (lag, 0), mode="replicate")
        pi = F.pad(i[..., :-lag], (lag, 0), mode="replicate")
        features.extend(adapted.features(r * pr + i * pi, i * pr - r * pi))
    torch.testing.assert_close(torch.cat(features, 1), expected)


@pytest.mark.parametrize("frontend", ["csp", "raw"])
@pytest.mark.parametrize("stride", [1, 4])
def test_variable_length_backward_and_checkpoint(frontend, stride):
    torch.manual_seed(2)
    c = MAMC2Config(backend="torch", frontend=frontend, stride=stride)
    model = MAMC2(c)
    parameter_count = sum(p.numel() for p in model.parameters())
    for length in [17, 129, 1025]:
        model.zero_grad(set_to_none=True)
        feature, y = model(torch.randn(2, 2, length), return_feature=True)
        assert y.shape == (2, 11) and feature.shape == (2, 64)
        F.cross_entropy(y, torch.tensor([1, 3])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert parameter_count == sum(p.numel() for p in model.parameters())
    other = MAMC2(c).eval()
    other.load_state_dict(model.state_dict())
    model.eval()
    x = torch.randn(1, 2, 75)
    torch.testing.assert_close(model(x), other(x))


def test_overfit_tiny_batch():
    torch.manual_seed(3)
    model = MAMC2(MAMC2Config(backend="torch", num_classes=2, n_layer=1, dropout=0))
    x = torch.randn(4, 2, 33)
    y = torch.tensor([0, 1, 0, 1])
    optimizer = torch.optim.AdamW(model.optimizer_groups(.01), lr=.01)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * .3


def test_dataset_and_disjoint_stratification(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, X=np.zeros((48, 33, 2)), Y=np.repeat([0, 1], 24), SNR=np.tile(np.repeat([-2, 2], 12), 2))
    data = IQDataset(path, length=17)
    assert data.x.shape == (48, 2, 17)
    splits = stratified_split(data, seed=7)
    assert len(set(np.concatenate(splits))) == len(data)
    assert sum(map(len, splits)) == len(data)
    for part in splits:
        assert set(data.y[part].tolist()) == {0, 1}
        assert set(data.snr[part].tolist()) == {-2., 2.}
    for a, b in zip(splits, stratified_split(data, seed=7)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="genuinely longer"):
        IQDataset(path, length=1024)


def test_cpu_official_fails_explicitly():
    # Avoid importing optional libraries; only validate the dispatch guard.
    model = MAMC2(MAMC2Config(backend="torch"))
    model.config = replace(model.config, backend="official")
    with pytest.raises(RuntimeError, match="needs CUDA"):
        model(torch.randn(1, 2, 33))


def test_no_decay_ssm_dynamics():
    model = MAMC2(MAMC2Config(backend="torch"))
    groups = model.optimizer_groups(.1)
    no_decay = {id(p) for p in groups[1]["params"]}
    for block in model.blocks:
        assert all(id(p) in no_decay for p in [block.mixer.A_log, block.mixer.D, block.mixer.dt_bias])


def test_upstream_parameter_layout_and_forward_parity(monkeypatch):
    """Use upstream forward/projections with an independent recurrent scan on CPU.

    Replaces CUDA-only scan/norm implementations, not upstream projection/split logic.
    This test does not establish Triton kernel compatibility or GPU speed.
    """
    try:
        import mamba_ssm.modules.mamba2 as upstream
    except (ImportError, OSError) as exc:
        pytest.skip(f"Optional upstream Mamba2 unavailable: {exc}")
    from mamc.mamba2 import GatedRMSNorm
    torch.manual_seed(13)
    official = upstream.Mamba2(d_model=32, d_state=32, headdim=32, chunk_size=16,
                               use_mem_eff_path=False)
    portable = TorchMamba2(d_model=32, d_state=32, headdim=32, chunk_size=16)
    portable.load_state_dict(official.state_dict(), strict=True)
    official.norm = GatedRMSNorm(64)
    official.norm.load_state_dict(portable.norm.state_dict())
    monkeypatch.setattr(upstream, "causal_conv1d_fn", None)
    def independent_scan(x, dt, A, B, C, **kwargs):
        assert kwargs["dt_softplus"]
        dt = F.softplus(dt + kwargs["dt_bias"])
        return recurrence(x, dt, A, B.squeeze(2), C.squeeze(2), kwargs["D"])
    monkeypatch.setattr(upstream, "mamba_chunk_scan_combined", independent_scan)
    x = torch.randn(2, 71, 32, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    actual, expected = portable(x), official(other_x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, other_x.grad, atol=3e-5, rtol=3e-4)
