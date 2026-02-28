# pyright: reportConstantRedefinition=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor

__all__ = ["PCA", "PCASymbols"]


def _normalize_device(device: object) -> torch.device:
    """device 指定を torch.device に正規化する（Fail Fast）。"""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    # bool は int のサブクラスなので先に弾く
    if isinstance(device, bool):
        raise TypeError("device に bool は指定できません。")
    if isinstance(device, int):
        if not torch.cuda.is_available():
            raise ValueError("device に int (CUDA index) を指定しましたが、CUDA が利用できません。")
        n = torch.cuda.device_count()
        if not (0 <= device < n):
            raise ValueError(f"CUDA index out of range: {device} (device_count={n})")
        return torch.device(f"cuda:{device}")
    raise TypeError(f"Unsupported device type: {type(device)}")


def _as_float_tensor_2d(
    x: object,
    *,
    dtype: torch.dtype | None,
    device: torch.device | str | int | None,
    name: str,
) -> Tensor:
    """入力を 2 次元の浮動小数点 Tensor として検証・整形する（Fail Fast）。

    Parameters
    ----------
    x : object
        入力。torch.Tensor である必要がある。
    dtype : torch.dtype | None
        指定時、dtype に変換する。
    device : torch.device | str | int | None
        指定時、device に移動する。int は CUDA index を意味する。
    name : str
        エラーメッセージに用いる名前。

    Returns
    -------
    X : torch.Tensor
        形状 (N, d) の 2 次元浮動小数点テンソル。

    Raises
    ------
    TypeError
        x が torch.Tensor でない、または浮動小数点でない場合。
    ValueError
        x が 2 次元でない、または空の場合。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} は torch.Tensor である必要があります: got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"{name} は2次元テンソル (N, d) である必要があります: shape={tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"{name} は空であってはなりません。")
    if not x.is_floating_point():
        raise TypeError(f"{name} は浮動小数点テンソルである必要があります: dtype={x.dtype}")

    out = x
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    if device is not None:
        dev = _normalize_device(device)
        if out.device != dev:
            out = out.to(device=dev)
    return out


def _safe_div(a: Tensor, b: Tensor, eps: float) -> Tensor:
    """ゼロ割を避けた安全な除算 a / max(b, eps)。"""
    return a / b.clamp_min(eps)


def _svd_rank(
    sigma: Tensor,
    *,
    N: int,
    d: int,
    rtol: float,
) -> int:
    """thin SVD の特異値列から数値ランクを推定する。

    Notes
    -----
    有限精度では rank 判定が不安定になりやすいので、次の形の閾値を採用する。

        tol = rtol * max(N, d) * sigma_max * eps(dtype)

    ここで eps(dtype) は機械イプシロン（torch.finfo）。
    """
    if sigma.numel() == 0:
        return 0
    if not sigma.is_floating_point():
        raise TypeError(f"sigma は浮動小数点である必要があります: dtype={sigma.dtype}")
    if N <= 0 or d <= 0:
        raise ValueError(f"N,d は正である必要があります: N={N}, d={d}")
    if rtol <= 0.0:
        raise ValueError(f"rtol は正である必要があります: rtol={rtol}")

    sigma_max = float(sigma.max())
    finfo = torch.finfo(sigma.dtype)
    tol = float(rtol) * float(max(N, d)) * sigma_max * float(finfo.eps)
    return int((sigma > tol).sum().item())


@dataclass(frozen=True)
class PCASymbols:
    """ドキュメントの記号をまとめて取り出すための薄いビュー。

    Notes
    -----
    - 各フィールドは `PCA` が保持する Tensor 参照をそのまま返す（複製しない）。
    - store_data=False の場合、X_c と Z は None になる。
    """
    mu: Tensor
    X_c: Tensor | None
    S: Tensor
    Q: Tensor
    Lambda: Tensor
    U: Tensor
    sigma: Tensor
    Sigma: Tensor
    V: Tensor
    W: Tensor
    Z: Tensor | None
    lambda_: Tensor
    EVR_within_k: Tensor
    EVR_total: Tensor


class PCA:
    r"""PCA（SVDベース）をドキュメント定義に忠実に実装した PyTorch クラス。

    定義（ドキュメント準拠）
    ------------------------
    - データ行列: :math:`\mathbf{X}\in\mathbb{R}^{N\times d}`
    - 全1ベクトル: :math:`\mathbf{1}_N\in\mathbb{R}^{N}`
    - 平均: :math:`\boldsymbol{\mu}=\frac{1}{N}\mathbf{X}^\top\mathbf{1}_N\in\mathbb{R}^{d}`
    - 中心化: :math:`\mathbf{X}_c=\mathbf{X}-\mathbf{1}_N\boldsymbol{\mu}^\top\in\mathbb{R}^{N\times d}`
    - 共分散（分母 N）: :math:`\mathbf{S}=\frac{1}{N}\mathbf{X}_c^\top\mathbf{X}_c\in\mathbb{R}^{d\times d}`
    - thin SVD: :math:`\mathbf{X}_c=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top`
      （:math:`\mathbf{U}\in\mathbb{R}^{N\times r}`, :math:`\boldsymbol{\Sigma}\in\mathbb{R}^{r\times r}`,
       :math:`\mathbf{V}\in\mathbb{R}^{d\times r}`）
    - 主成分方向: :math:`\mathbf{W}=\mathbf{V}_k\in\mathbb{R}^{d\times k}`
    - スコア: :math:`\mathbf{Z}=\mathbf{X}_c\mathbf{W}\in\mathbb{R}^{N\times k}=\mathbf{U}_k\boldsymbol{\Sigma}_k`
    - 説明分散（分母 N）: :math:`\lambda_m=\sigma_m^2/N`
    - 説明分散比（保持k内で正規化）: :math:`\mathrm{EVR}^{(k)}_m=\lambda_m/\sum_{j=1}^{k}\lambda_j`
    - 説明分散比（総分散で正規化）: :math:`\mathrm{EVR}^{(\mathrm{total})}_m=\lambda_m/\sum_{j=1}^{r}\lambda_j`
    - whitening（任意）:
      :math:`\mathbf{Z}^{\mathrm{white}}=\mathbf{Z}\mathrm{diag}(\lambda_1^{-1/2},\dots,\lambda_k^{-1/2})`

    Parameters
    ----------
    n_components : int
        主成分数 k。`1 <= k <= min(N, d)`。
    center : bool, default=True
        True の場合、中心化を行う。False の場合は μ=0 とみなし、X_c=X を用いる。
    whiten : bool, default=False
        True の場合、transform() のデフォルト出力を whitening して返す。
    store_data : bool, default=True
        True の場合、X_c と学習データの（非whiten）Z を保持する。
    eps : float, default=1e-12
        ゼロ割回避用の小定数。
    dtype : torch.dtype | None
        指定時、内部 dtype に揃える。
    device : torch.device | str | int | None
        指定時、内部 device に揃える。int は CUDA index を意味する。
    rank_rtol : float, default=1e-7
        数値ランク推定の相対閾値係数。
    differentiable : bool, default=False
        True の場合、fit() を含む計算を no_grad で包まない（=勾配グラフを保持し得る）。
        通常の PCA では False 推奨。

    Attributes
    ----------
    k : int
        主成分数。
    N, d, r : int | None
        サンプル数 / 特徴次元 / 数値ランク。
    mu_, X_c_ : torch.Tensor | None
        μ と X_c（store_data=True のとき X_c_ を保持）。
    U_, sigma_, Sigma_, V_ : torch.Tensor | None
        thin SVD の U, σ（ベクトル）, Σ（対角行列）, V。
    W_ : torch.Tensor | None
        主成分方向 W=V_k。
    Z_ : torch.Tensor | None
        学習データのスコア Z（非whiten、store_data=True のとき保持）。
    lambda_, EVR_within_k_, EVR_total_ : torch.Tensor | None
        説明分散と説明分散比（k内正規化 / 総分散正規化）。
    """
    # --- public-ish config ---
    k: int
    center: bool
    whiten: bool
    store_data: bool
    eps: float
    rank_rtol: float
    cov_ddof: Literal[0, 1]
    differentiable: bool

    # --- learned ---
    N: int | None
    d: int | None
    r: int | None

    mu_: Tensor | None
    X_c_: Tensor | None

    U_: Tensor | None
    sigma_: Tensor | None
    Sigma_: Tensor | None
    V_: Tensor | None

    W_: Tensor | None
    Z_: Tensor | None

    lambda_: Tensor | None
    EVR_within_k_: Tensor | None
    EVR_total_: Tensor | None

    def __init__(
        self,
        n_components: int,
        *,
        center: bool = True,
        whiten: bool = False,
        store_data: bool = True,
        eps: float = 1e-12,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        rank_rtol: float = 1e-7,
        cov_ddof: Literal[0, 1] = 1,
        differentiable: bool = False,
    ) -> None:
        if int(n_components) < 1:
            raise ValueError(f"n_components は 1 以上である必要があります: got {n_components}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {eps}")
        if float(rank_rtol) <= 0.0:
            raise ValueError(f"rank_rtol は正である必要があります: got {rank_rtol}")
        if cov_ddof not in (0, 1):
            raise ValueError(f"cov_ddof は 0 または 1 である必要があります: got {cov_ddof}")

        self.k = int(n_components)
        self.center = bool(center)
        self.whiten = bool(whiten)
        self.store_data = bool(store_data)
        self.eps = float(eps)
        self.rank_rtol = float(rank_rtol)
        self.cov_ddof = cov_ddof
        self.differentiable = bool(differentiable)

        self._dtype: torch.dtype | None = dtype
        self._device: torch.device | None = _normalize_device(device) if device is not None else None

        self.N = None
        self.d = None
        self.r = None

        self.mu_ = None
        self.X_c_ = None

        self.U_ = None
        self.sigma_ = None
        self.Sigma_ = None
        self.V_ = None

        self.W_ = None
        self.Z_ = None

        self.lambda_ = None
        self.EVR_within_k_ = None
        self.EVR_total_ = None

        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("PCAは未学習です。fit() を先に呼んでください。")

    # ---- symbol-faithful properties ----
    @property
    def mu(self) -> Tensor:
        self._require_fit()
        assert self.mu_ is not None
        return self.mu_

    @property
    def X_c(self) -> Tensor:
        self._require_fit()
        if self.X_c_ is None:
            raise RuntimeError("store_data=False のため X_c は保持していません。")
        return self.X_c_

    @property
    def U(self) -> Tensor:
        self._require_fit()
        assert self.U_ is not None
        return self.U_

    @property
    def sigma(self) -> Tensor:
        self._require_fit()
        assert self.sigma_ is not None
        return self.sigma_

    @property
    def Sigma(self) -> Tensor:
        self._require_fit()
        assert self.Sigma_ is not None
        return self.Sigma_

    @property
    def V(self) -> Tensor:
        self._require_fit()
        assert self.V_ is not None
        return self.V_

    @property
    def W(self) -> Tensor:
        self._require_fit()
        assert self.W_ is not None
        return self.W_

    @property
    def Z(self) -> Tensor:
        self._require_fit()
        if self.Z_ is None:
            raise RuntimeError(
                "store_data=False のため学習データZは保持していません。transform(X) を使用してください。"
            )
        return self.Z_

    @property
    def lambda_k(self) -> Tensor:
        """保持 k 成分の説明分散 λ（shape: (k,)）を返す。"""
        self._require_fit()
        assert self.lambda_ is not None
        return self.lambda_

    @property
    def EVR_within_k(self) -> Tensor:
        """保持 k 内で正規化された説明分散比（shape: (k,)）。"""
        self._require_fit()
        assert self.EVR_within_k_ is not None
        return self.EVR_within_k_

    @property
    def EVR_total(self) -> Tensor:
        """総分散（rank=r まで）で正規化された説明分散比（shape: (k,)）。"""
        self._require_fit()
        assert self.EVR_total_ is not None
        return self.EVR_total_

    @property
    def S(self) -> Tensor:
        """共分散 S を返す（SVD から再構成、分母 N-cov_ddof）。"""
        self._require_fit()
        assert self.V_ is not None
        assert self.sigma_ is not None
        assert self.N is not None

        denom = float(self.N - int(self.cov_ddof))
        scale = (self.sigma_ * self.sigma_) / denom  # (r,)
        return (self.V_ * scale.unsqueeze(0)) @ self.V_.T  # (d, d)

    @property
    def Q(self) -> Tensor:
        """固有ベクトル行列 Q（thin SVD では Q=V）。"""
        self._require_fit()
        assert self.V_ is not None
        return self.V_

    @property
    def Lambda(self) -> Tensor:
        """固有値対角行列 Λ（thin SVD では r 次）。"""
        self._require_fit()
        assert self.sigma_ is not None
        assert self.N is not None
        denom = float(self.N - int(self.cov_ddof))
        lam = (self.sigma_ * self.sigma_) / denom
        return torch.diag(lam)

    @property
    def symbols(self) -> PCASymbols:
        """ドキュメント記号一式をまとめて返す。"""
        self._require_fit()
        assert self.mu_ is not None
        assert self.U_ is not None
        assert self.sigma_ is not None
        assert self.Sigma_ is not None
        assert self.V_ is not None
        assert self.W_ is not None
        assert self.lambda_ is not None
        assert self.EVR_within_k_ is not None
        assert self.EVR_total_ is not None

        return PCASymbols(
            mu=self.mu_,
            X_c=self.X_c_ if self.store_data else None,
            S=self.S,
            Q=self.Q,
            Lambda=self.Lambda,
            U=self.U_,
            sigma=self.sigma_,
            Sigma=self.Sigma_,
            V=self.V_,
            W=self.W_,
            Z=self.Z_ if self.store_data else None,
            lambda_=self.lambda_,
            EVR_within_k=self.EVR_within_k_,
            EVR_total=self.EVR_total_,
        )

    # ---- core API ----
    def fit(self, X: Tensor) -> PCA:
        """中心化と thin SVD により PCA を学習する。

        Notes
        -----
        differentiable=False（デフォルト）の場合、fit は no_grad で実行される。
        """
        ctx = torch.enable_grad() if self.differentiable else torch.no_grad()
        with ctx:
            X_tensor = _as_float_tensor_2d(X, dtype=self._dtype, device=self._device, name="X")
            N, d = int(X_tensor.shape[0]), int(X_tensor.shape[1])

            denom = float(N - int(self.cov_ddof))
            if denom <= 0.0:
                raise ValueError(f"cov_ddof={self.cov_ddof} のため分母 N-cov_ddof が非正です: N={N}")

            if self.k > min(N, d):
                raise ValueError(
                    f"n_components(k)={self.k} は min(N,d)={min(N, d)} 以下である必要があります。"
                )

            if self.center:
                # μ = mean over rows (shape: (d,))
                mu = X_tensor.mean(dim=0)
                X_c = X_tensor - mu.unsqueeze(0)
            else:
                mu = torch.zeros(d, dtype=X_tensor.dtype, device=X_tensor.device)
                X_c = X_tensor

            # thin SVD: X_c = U diag(σ) V^T
            U_full, sigma_full, Vh_full = torch.linalg.svd(X_c, full_matrices=False)

            r = _svd_rank(sigma_full, N=N, d=d, rtol=self.rank_rtol)
            if r == 0:
                raise ValueError(
                    "数値ランクが 0 です。入力が定数（中心化後ゼロ）である可能性があります。"
                )

            # r へトリム
            U = U_full[:, :r].contiguous()          # (N, r)
            sigma = sigma_full[:r].contiguous()     # (r,)
            Vh = Vh_full[:r, :].contiguous()        # (r, d)
            V = Vh.T.contiguous()                   # (d, r)
            Sigma = torch.diag(sigma)               # (r, r)

            if self.k > r:
                raise ValueError(
                    f"k={self.k} は 数値ランク r={r} 以下である必要があります（rank_rtol={self.rank_rtol}）。"
                )

            W = V[:, : self.k].contiguous()         # (d, k)

            # λ_m = σ_m^2 / (N - cov_ddof)  （m=1..k）
            sigma_k = sigma[: self.k]
            lambda_k = (sigma_k * sigma_k) / denom  # (k,)

            # EVR: within-k & total (rank-r)
            sum_k = lambda_k.sum()
            EVR_within_k = _safe_div(lambda_k, sum_k, eps=self.eps)

            lambda_r = (sigma * sigma) / denom  # (r,)
            sum_r = lambda_r.sum()
            EVR_total = _safe_div(lambda_k, sum_r, eps=self.eps)

            # 保存
            self.N = N
            self.d = d
            self.r = r

            self.mu_ = mu

            self.U_ = U
            self.sigma_ = sigma
            self.Sigma_ = Sigma
            self.V_ = V

            self.W_ = W

            self.lambda_ = lambda_k
            self.EVR_within_k_ = EVR_within_k
            self.EVR_total_ = EVR_total

            if self.store_data:
                self.X_c_ = X_c
                # Z = U_k Σ_k
                U_k = U[:, : self.k]                           # (N, k)
                Z = U_k * sigma_k.unsqueeze(0)                 # (N, k)
                self.Z_ = Z
            else:
                self.X_c_ = None
                self.Z_ = None

            self._fitted = True
            return self

    def transform(self, X: Tensor, *, whiten: bool | None = None) -> Tensor:
        """データを上位 k 主成分へ射影し、スコア Z（または whitening 後）を返す。

        Parameters
        ----------
        X : torch.Tensor
            形状 (N, d) の入力データ。
        whiten : bool | None, default=None
            - None の場合、コンストラクタの self.whiten に従う。
            - True の場合、Z を whitening して返す。
            - False の場合、Z（非whiten）を返す。

        Returns
        -------
        Z : torch.Tensor
            形状 (N, k) のスコア。
        """
        self._require_fit()
        assert self.mu_ is not None
        assert self.W_ is not None
        assert self.d is not None

        X_tensor = _as_float_tensor_2d(X, dtype=self.mu_.dtype, device=self.mu_.device, name="X")
        if int(X_tensor.shape[1]) != int(self.d):
            raise ValueError(f"入力 d={int(X_tensor.shape[1])} が学習時 d={int(self.d)} と一致しません。")

        if self.center:
            X_c = X_tensor - self.mu_.unsqueeze(0)
        else:
            X_c = X_tensor

        Z = X_c @ self.W_  # (N, k)

        do_whiten = self.whiten if whiten is None else bool(whiten)
        if do_whiten:
            assert self.lambda_ is not None
            Z = self.whiten_scores(Z)
        return Z

    def fit_transform(self, X: Tensor, *, whiten: bool | None = None) -> Tensor:
        """fit(X) の後に transform(X) を実行してスコアを返す。

        Notes
        -----
        store_data=True かつ whiten=False の場合、fit() で計算済みの Z_ を返す（再計算しない）。
        """
        self.fit(X)

        do_whiten = self.whiten if whiten is None else bool(whiten)
        if self.store_data and (not do_whiten):
            # fit() 内で非whitenの Z を保存済み
            return self.Z

        return self.transform(X, whiten=do_whiten)

    def whiten_scores(self, Z: Tensor) -> Tensor:
        """Z を whitening する（Z_white = Z / sqrt(lambda)）。

        Parameters
        ----------
        Z : torch.Tensor
            形状 (N, k) の非whitenスコア。

        Returns
        -------
        Z_white : torch.Tensor
            形状 (N, k) の whitened スコア。
        """
        self._require_fit()
        assert self.lambda_ is not None

        Z_tensor = _as_float_tensor_2d(Z, dtype=self.lambda_.dtype, device=self.lambda_.device, name="Z")
        if int(Z_tensor.shape[1]) != int(self.k):
            raise ValueError(f"Z の列数={int(Z_tensor.shape[1])} は k={int(self.k)} と一致する必要があります。")

        denom = torch.sqrt(self.lambda_).unsqueeze(0)  # (1, k)
        return _safe_div(Z_tensor, denom, eps=self.eps)

    def unwhiten_scores(self, Z_white: Tensor) -> Tensor:
        """whitened スコアを unwhiten する（Z = Z_white * sqrt(lambda)）。

        Parameters
        ----------
        Z_white : torch.Tensor
            形状 (N, k) の whitened スコア。

        Returns
        -------
        Z : torch.Tensor
            形状 (N, k) の非whitenスコア。
        """
        self._require_fit()
        assert self.lambda_ is not None

        Z_tensor = _as_float_tensor_2d(Z_white, dtype=self.lambda_.dtype, device=self.lambda_.device, name="Z_white")
        if int(Z_tensor.shape[1]) != int(self.k):
            raise ValueError(
                f"Z_white の列数={int(Z_tensor.shape[1])} は k={int(self.k)} と一致する必要があります。"
            )

        scale = torch.sqrt(self.lambda_).unsqueeze(0)  # (1, k)
        return Z_tensor * scale

    def inverse_transform(self, Z: Tensor, *, whitened: bool | None = None) -> Tensor:
        """スコア Z から元の特徴空間へ再構成する。

        Parameters
        ----------
        Z : torch.Tensor
            形状 (N, k) のスコア。デフォルトでは self.whiten に従う。
        whitened : bool | None, default=None
            - None の場合、self.whiten を参照して「入力 Z が whitened か」を解釈する。
            - True の場合、Z は whitened とみなし unwhiten してから逆変換する。
            - False の場合、Z は非whiten とみなす。

        Returns
        -------
        X_hat : torch.Tensor
            形状 (N, d) の再構成。

        Notes
        -----
        - X_hat_c = Z W^T
        - center=True の場合: X_hat = X_hat_c + 1_N μ^T
        """
        self._require_fit()
        assert self.W_ is not None
        assert self.mu_ is not None
        assert self.d is not None
        assert self.lambda_ is not None

        Z_tensor = _as_float_tensor_2d(Z, dtype=self.mu_.dtype, device=self.mu_.device, name="Z")
        if int(Z_tensor.shape[1]) != int(self.k):
            raise ValueError(f"Z の列数={int(Z_tensor.shape[1])} は k={int(self.k)} と一致する必要があります。")

        is_whitened = self.whiten if whitened is None else bool(whitened)
        if is_whitened:
            Z_tensor = self.unwhiten_scores(Z_tensor)

        X_hat_c = Z_tensor @ self.W_.T  # (N, d)
        if not self.center:
            return X_hat_c

        return X_hat_c + self.mu_.unsqueeze(0)

    def explained_variance_ratio(self, *, mode: Literal["within_k", "total"] = "within_k") -> Tensor:
        """説明分散比を返す。

        Parameters
        ----------
        mode : {"within_k", "total"}, default="within_k"
            - "within_k": 保持k内で正規化
            - "total": rank=r の総分散で正規化

        Returns
        -------
        evr : torch.Tensor
            形状 (k,) の説明分散比。
        """
        self._require_fit()
        if mode == "within_k":
            return self.EVR_within_k
        if mode == "total":
            return self.EVR_total
        raise ValueError(f"Unsupported mode: {mode}")