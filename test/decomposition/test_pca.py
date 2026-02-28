from __future__ import annotations

import pytest
import torch

from decomposition.pca import PCA, _normalize_device


def _make_random_X(*, N: int = 64, d: int = 10, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    torch.manual_seed(0)
    X = torch.randn(N, d, dtype=dtype)
    # 平均が極端に偏らないように軽くシフト
    X = X + 0.1 * torch.randn(1, d, dtype=dtype)
    return X


def _make_X(
    *,
    N: int = 32,
    d: int = 12,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    torch.manual_seed(0)
    X = torch.randn(N, d, dtype=dtype, device=device)
    X.requires_grad_(requires_grad)
    return X


def _cov_Nm1(X: torch.Tensor) -> torch.Tensor:
    """分母 (N-1) の共分散: (X^T X) / (N-1)"""
    N = X.shape[0]
    if N <= 1:
        raise ValueError("N must be >= 2 for ddof=1 covariance.")
    return (X.T @ X) / float(N - 1)


# ==============================
# PCA: core behavior
# ==============================

def test_fit_shapes_and_basic_identities() -> None:
    X = _make_random_X(N=50, d=12)
    k = 5
    pca = PCA(n_components=k, center=True, whiten=False, store_data=True, dtype=X.dtype).fit(X)

    assert pca.fitted is True
    assert pca.N == 50
    assert pca.d == 12
    assert pca.r is not None
    assert 1 <= pca.r <= min(pca.N, pca.d)

    # shapes
    assert pca.mu.shape == (12,)
    assert pca.X_c.shape == (50, 12)
    assert pca.U.shape == (50, pca.r)
    assert pca.sigma.shape == (pca.r,)
    assert pca.Sigma.shape == (pca.r, pca.r)
    assert pca.V.shape == (12, pca.r)
    assert pca.W.shape == (12, k)
    assert pca.Z.shape == (50, k)
    assert pca.lambda_k.shape == (k,)
    assert pca.EVR_within_k.shape == (k,)
    assert pca.EVR_total.shape == (k,)

    # X_c = X - 1 mu^T
    X_c_expected = X - pca.mu.unsqueeze(0)
    assert torch.allclose(pca.X_c, X_c_expected, rtol=1e-10, atol=1e-10)

    # Z = X_c W
    Z_expected = pca.X_c @ pca.W
    assert torch.allclose(pca.Z, Z_expected, rtol=1e-10, atol=1e-10)

    # U, V は（thin）直交列ベクトル: U^T U = I_r, V^T V = I_r
    I_r = torch.eye(pca.r, dtype=X.dtype)
    assert torch.allclose(pca.U.T @ pca.U, I_r, rtol=1e-8, atol=1e-8)
    assert torch.allclose(pca.V.T @ pca.V, I_r, rtol=1e-8, atol=1e-8)

    # W^T W = I_k
    I_k = torch.eye(k, dtype=X.dtype)
    assert torch.allclose(pca.W.T @ pca.W, I_k, rtol=1e-8, atol=1e-8)

    # lambda_k = sigma_k^2 / (N-1)   (cov_ddof default = 1)
    sigma_k = pca.sigma[:k]
    lambda_expected = (sigma_k * sigma_k) / float(pca.N - 1)
    assert torch.allclose(pca.lambda_k, lambda_expected, rtol=1e-12, atol=1e-12)

    # EVR_within_k は総和1
    assert torch.allclose(
        pca.EVR_within_k.sum(),
        torch.tensor(1.0, dtype=X.dtype),
        rtol=1e-12,
        atol=1e-12,
    )


def test_covariance_reconstruction_matches_definition() -> None:
    X = _make_random_X(N=80, d=20)
    pca = PCA(n_components=10, center=True, dtype=X.dtype).fit(X)

    # S = X_c^T X_c / (N-1)  (cov_ddof default = 1)
    S_expected = _cov_Nm1(pca.X_c)
    assert torch.allclose(pca.S, S_expected, rtol=1e-8, atol=1e-8)


def test_transform_equals_fit_transform_when_not_whiten_and_store_data_true() -> None:
    X = _make_random_X(N=40, d=15)
    pca = PCA(n_components=6, center=True, whiten=False, store_data=True, dtype=X.dtype)

    Z_fit_transform = pca.fit_transform(X)
    Z_transform = pca.transform(X)

    assert torch.allclose(Z_fit_transform, Z_transform, rtol=1e-10, atol=1e-10)
    assert torch.allclose(Z_fit_transform, pca.Z, rtol=1e-10, atol=1e-10)


def test_whiten_unwhiten_roundtrip_and_whitened_cov_is_identity() -> None:
    X = _make_random_X(N=100, d=8)
    k = 4
    pca = PCA(n_components=k, center=True, whiten=False, store_data=True, dtype=X.dtype).fit(X)

    Z = pca.transform(X, whiten=False)
    Z_white = pca.whiten_scores(Z)
    Z_back = pca.unwhiten_scores(Z_white)

    assert torch.allclose(Z, Z_back, rtol=1e-10, atol=1e-10)

    # whitening 後は (Z_white^T Z_white)/(N-1) ≈ I_k になる（ddof=1 規約）
    Cw = (Z_white.T @ Z_white) / float(pca.N - 1)
    I_k = torch.eye(k, dtype=X.dtype)
    assert torch.allclose(Cw, I_k, rtol=1e-6, atol=1e-6)


def test_inverse_transform_reconstructs_centered_part() -> None:
    X = _make_random_X(N=60, d=18)
    k = 7
    pca = PCA(n_components=k, center=True, whiten=False, store_data=True, dtype=X.dtype).fit(X)

    Z = pca.transform(X, whiten=False)
    X_hat = pca.inverse_transform(Z, whitened=False)

    # 再構成は rank-k 射影: X_hat = (X_c W W^T) + mu
    X_c = X - pca.mu.unsqueeze(0)
    X_hat_expected = (X_c @ pca.W @ pca.W.T) + pca.mu.unsqueeze(0)

    assert torch.allclose(X_hat, X_hat_expected, rtol=1e-8, atol=1e-8)


def test_center_false_sets_mu_zero_and_Xc_equals_X() -> None:
    X = _make_random_X(N=30, d=9)
    pca = PCA(n_components=3, center=False, dtype=X.dtype).fit(X)

    assert torch.allclose(pca.mu, torch.zeros_like(pca.mu))
    assert torch.allclose(pca.X_c, X)

    Z = pca.transform(X)
    assert torch.allclose(Z, X @ pca.W, rtol=1e-10, atol=1e-10)


def test_store_data_false_disallows_access_to_Xc_and_Z_properties() -> None:
    X = _make_random_X(N=40, d=10)
    pca = PCA(n_components=3, store_data=False, dtype=X.dtype).fit(X)

    with pytest.raises(RuntimeError):
        _ = pca.X_c

    with pytest.raises(RuntimeError):
        _ = pca.Z

    Z = pca.transform(X)
    assert Z.shape == (40, 3)


def test_evr_total_sums_to_one_when_k_equals_rank() -> None:
    X = _make_random_X(N=40, d=25)
    r = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=r, center=True, dtype=X.dtype, rank_rtol=1e-12).fit(X)

    assert pca.r == r
    assert torch.allclose(
        pca.EVR_total.sum(),
        torch.tensor(1.0, dtype=X.dtype),
        rtol=1e-8,
        atol=1e-8,
    )


# ==============================
# PCA: Fail Fast / validation
# ==============================

def test_fit_requires_float_2d_tensor() -> None:
    pca = PCA(n_components=2)

    with pytest.raises(TypeError):
        pca.fit([[1.0, 2.0]])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        pca.fit(torch.randn(10))  # 1D

    with pytest.raises(ValueError):
        pca.fit(torch.randn(0, 3))  # empty

    with pytest.raises(TypeError):
        pca.fit(torch.randint(0, 10, (5, 3), dtype=torch.int64))  # non-float


def test_n_components_constraints() -> None:
    with pytest.raises(ValueError):
        _ = PCA(n_components=0)

    X = _make_random_X(N=5, d=4)
    pca = PCA(n_components=10)
    with pytest.raises(ValueError):
        pca.fit(X)  # k > min(N,d)


def test_rank_zero_raises_on_constant_input() -> None:
    X = torch.ones(20, 5, dtype=torch.float64)
    pca = PCA(n_components=1, center=True, dtype=X.dtype)

    with pytest.raises(ValueError):
        pca.fit(X)


def test_transform_requires_fit_and_matching_d() -> None:
    X = _make_random_X(N=20, d=6)
    pca = PCA(n_components=2)

    with pytest.raises(RuntimeError):
        _ = pca.transform(X)

    pca.fit(X)
    with pytest.raises(ValueError):
        _ = pca.transform(_make_random_X(N=10, d=7))


def test_inverse_transform_shape_checks() -> None:
    X = _make_random_X(N=20, d=6)
    pca = PCA(n_components=3, dtype=X.dtype).fit(X)

    with pytest.raises(ValueError):
        _ = pca.inverse_transform(torch.randn(10, 2, dtype=X.dtype))


# ==============================
# _normalize_device(): device=int(CUDA index)
# ==============================

def test_normalize_device_rejects_bool() -> None:
    with pytest.raises(TypeError):
        _ = _normalize_device(True)


def test_normalize_device_accepts_str_and_torch_device() -> None:
    d0 = _normalize_device("cpu")
    assert isinstance(d0, torch.device)
    assert d0.type == "cpu"

    d1 = _normalize_device(torch.device("cpu"))
    assert isinstance(d1, torch.device)
    assert d1.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_normalize_device_int_cuda_index_maps_to_cuda_device() -> None:
    dev = _normalize_device(0)
    assert isinstance(dev, torch.device)
    assert dev.type == "cuda"
    assert dev.index == 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test expects CUDA to be unavailable")
def test_normalize_device_int_raises_when_cuda_unavailable() -> None:
    with pytest.raises(ValueError):
        _ = _normalize_device(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_normalize_device_int_out_of_range_raises() -> None:
    n = torch.cuda.device_count()
    with pytest.raises(ValueError):
        _ = _normalize_device(n)


# ==============================
# differentiable=True: gradient behavior
# ==============================

def test_fit_default_is_nondifferentiable_detaches_W() -> None:
    X = _make_X(N=40, d=10, requires_grad=True)
    pca = PCA(n_components=5, center=True, whiten=False, store_data=True, dtype=X.dtype)
    pca.fit(X)

    W = pca.W
    assert W.requires_grad is False

    loss = (W * W).sum()
    with pytest.raises(RuntimeError):
        loss.backward()


def test_fit_differentiable_true_keeps_graph_through_svd_to_X() -> None:
    X = _make_X(N=50, d=12, requires_grad=True)
    pca = PCA(
        n_components=6,
        center=True,
        whiten=False,
        store_data=True,
        dtype=X.dtype,
        differentiable=True,
    ).fit(X)

    W = pca.W
    assert W.requires_grad is True

    loss = (W * W).sum()
    loss.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_transform_grad_flows_to_X_even_if_fit_nondifferentiable() -> None:
    X = _make_X(N=30, d=9, requires_grad=True)
    pca = PCA(n_components=4, center=True, whiten=False, store_data=True, dtype=X.dtype).fit(X.detach())

    Z = pca.transform(X)
    assert Z.requires_grad is True

    loss = (Z * Z).mean()
    loss.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_int_moves_internal_tensors_to_cuda() -> None:
    X_cpu = _make_X(N=20, d=8, requires_grad=False, device="cpu")
    pca = PCA(n_components=3, device=0, dtype=X_cpu.dtype).fit(X_cpu)

    assert pca.mu.device.type == "cuda"
    assert pca.W.device.type == "cuda"

    Z = pca.transform(X_cpu)
    assert Z.device.type == "cuda"