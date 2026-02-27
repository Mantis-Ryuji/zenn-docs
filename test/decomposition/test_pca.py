from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from decomposition.pca import PCA


@pytest.fixture(scope="module")
def rng_seed() -> int:
    return 1234


def _device_candidates() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _device_candidates())
def test_fit_transform_shapes_and_dtypes(device: str, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(200, 64, device=device, dtype=torch.float32)

    pca = PCA(n_components=10, center=True, whiten=False)
    Z = pca.fit_transform(X)

    assert Z.shape == (200, 10)
    assert Z.device.type == torch.device(device).type
    assert Z.dtype == torch.float32

    # attribute shapes
    assert pca.mean_.shape == (64,)
    assert pca.loadings_.shape == (64, 10)
    assert pca.singular_values_.shape == (10,)
    assert pca.explained_variance_.shape == (10,)
    assert pca.explained_variance_ratio_.shape == (10,)
    assert pca.n_samples_ == 200
    assert pca.fitted_ is True


@pytest.mark.parametrize("device", _device_candidates())
def test_loadings_are_orthonormal(device: str, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(128, 50, device=device)

    k = 20
    pca = PCA(n_components=k, center=True).fit(X)
    W = pca.loadings  # (d, k)

    # W^T W == I_k
    I = torch.eye(k, device=W.device, dtype=W.dtype)
    WT_W = W.T @ W
    assert torch.allclose(WT_W, I, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _device_candidates())
def test_transform_matches_definition(device: str, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(100, 32, device=device)

    pca = PCA(n_components=8, center=True).fit(X)
    Z1 = pca.transform(X)

    # Definition: Z = (X - mean) @ loadings
    Xc = X - pca.mean_
    Z2 = Xc @ pca.loadings_
    assert torch.allclose(Z1, Z2, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _device_candidates())
def test_center_false_means_no_centering(device: str, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(120, 40, device=device)

    pca = PCA(n_components=5, center=False).fit(X)
    assert torch.allclose(pca.mean_, torch.zeros(40, device=device, dtype=X.dtype))

    Z1 = pca.transform(X)
    Z2 = X @ pca.loadings_
    assert torch.allclose(Z1, Z2, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _device_candidates())
def test_whiten_produces_unit_variance_per_component(device: str, rng_seed: int) -> None:
    # Whitening definition here: Z_white = Z / sqrt(lambda)
    # where lambda_i = sigma_i^2 / N  (denominator N convention)
    torch.manual_seed(rng_seed)
    X = torch.randn(2000, 30, device=device)

    pca = PCA(n_components=10, center=True, whiten=True).fit(X)
    Z = pca.transform(X)  # whitened

    # Empirical variance over samples should be close to 1 for each component.
    # Use unbiased=False to match denominator N convention.
    var = Z.var(dim=0, unbiased=False)
    ones = torch.ones_like(var)
    assert torch.allclose(var, ones, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", _device_candidates())
def test_explained_variance_ratio_sums_to_one(device: str, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(300, 80, device=device)

    pca = PCA(n_components=12).fit(X)
    s = float(pca.explained_variance_ratio_.sum().item())
    assert math.isfinite(s)
    assert abs(s - 1.0) < 1e-6


def test_save_and_load_roundtrip_cpu(tmp_path: Path, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(150, 60, device="cpu")

    pca = PCA(n_components=15, center=True, whiten=False).fit(X)

    out = tmp_path / "pca.pt"
    pca.save(out)

    pca2 = PCA.load(out, map_location="cpu")

    # Parameters equal
    assert torch.allclose(pca.mean_, pca2.mean_)
    assert torch.allclose(pca.loadings_, pca2.loadings_)
    assert torch.allclose(pca.singular_values_, pca2.singular_values_)
    assert torch.allclose(pca.explained_variance_, pca2.explained_variance_)
    assert torch.allclose(pca.explained_variance_ratio_, pca2.explained_variance_ratio_)
    assert pca.n_samples_ == pca2.n_samples_
    assert pca.center == pca2.center
    assert pca.whiten == pca2.whiten

    # Transform consistency
    Z1 = pca.transform(X)
    Z2 = pca2.transform(X)
    assert torch.allclose(Z1, Z2, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_save_load_to_cuda(tmp_path: Path, rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(256, 64, device="cuda")

    pca = PCA(n_components=16).fit(X)
    out = tmp_path / "pca.pt"
    pca.save(out)

    # load to cpu then move to cuda via device=
    pca2 = PCA.load(out, map_location="cpu", device="cuda")
    assert pca2.mean_.device.type == "cuda"
    assert pca2.loadings_.device.type == "cuda"

    Z1 = pca.transform(X)
    Z2 = pca2.transform(X)
    assert torch.allclose(Z1, Z2, atol=1e-5, rtol=1e-5)


def test_fail_fast_invalid_inputs() -> None:
    pca = PCA(n_components=2)

    # Not a tensor
    with pytest.raises(TypeError):
        pca.fit([[1.0, 2.0]])  # type: ignore[arg-type]

    # Not float
    with pytest.raises(TypeError):
        pca.fit(torch.ones(10, 3, dtype=torch.int64))

    # Not 2D
    with pytest.raises(ValueError):
        pca.fit(torch.randn(10))

    # Empty
    with pytest.raises(ValueError):
        pca.fit(torch.empty(0, 3))

    # n_components too large
    with pytest.raises(ValueError):
        PCA(n_components=999).fit(torch.randn(10, 3))


def test_transform_dimension_mismatch_raises(rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    X = torch.randn(100, 20)
    pca = PCA(n_components=5).fit(X)

    with pytest.raises(ValueError):
        pca.transform(torch.randn(10, 21))