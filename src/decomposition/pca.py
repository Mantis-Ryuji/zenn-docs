from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TypeAlias, Union

import torch

Tensor: TypeAlias = torch.Tensor
PathLike: TypeAlias = Union[str, Path]


def _as_tensor_2d(x: Any, *, dtype: Optional[torch.dtype], device: Optional[torch.device]) -> Tensor:
    """入力を2次元float Tensorとして検証・整形する（Fail Fast）。

    Parameters
    ----------
    x : Any
        入力。torch.Tensor である必要がある。
    dtype : torch.dtype, optional
        指定時、dtype に変換する。
    device : torch.device, optional
        指定時、device に移動する。

    Returns
    -------
    X : torch.Tensor
        形状 (N, d) の2次元浮動小数点テンソル。

    Raises
    ------
    TypeError
        x が torch.Tensor でない、または浮動小数点でない場合。
    ValueError
        x が2次元でない、または空の場合。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"X は torch.Tensor である必要があります: got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"X は2次元テンソル (N, d) である必要があります: shape={tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError("X は空であってはなりません。")
    if not x.is_floating_point():
        raise TypeError(f"X は浮動小数点テンソルである必要があります: dtype={x.dtype}")

    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype=dtype)
    if device is not None and x.device != device:
        x = x.to(device=device)
    return x


def _safe_div(a: Tensor, b: Tensor, eps: float) -> Tensor:
    """ゼロ割を避けた安全な除算 a / max(b, eps)。"""
    return a / b.clamp_min(eps)


@dataclass(frozen=True)
class PCAState:
    """PCAの保存用状態（torch.saveのpayloadに載せる）。"""
    n_components: int
    center: bool
    whiten: bool
    eps: float
    mean: Tensor
    loadings: Tensor
    singular_values: Tensor
    explained_variance: Tensor
    explained_variance_ratio: Tensor
    n_samples: int
    dtype: str
    device: str


class PCA:
    r"""PyTorch実装のPCA（SVDベース、GPU対応）。

    本クラスは次の定式化に厳密に従う。

    **中心化**
    - 入力行列を :math:`\mathbf{X}\in\mathbb{R}^{N\times d}` とする。
    - 特徴量平均 :math:`\boldsymbol{\mu}\in\mathbb{R}^{d}` を用いて
      :math:`\mathbf{X}_c = \mathbf{X} - \mathbf{1}\boldsymbol{\mu}^{\top}` とする。

    **（打ち切り）SVD**
    - :math:`\mathbf{X}_c \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^{\top}`

    **主軸（loadings）とスコア（scores）**
    - loadings（主成分方向）: :math:`\mathbf{W}=\mathbf{V}_k \in \mathbb{R}^{d\times k}`
    - scores（主成分得点）: :math:`\mathbf{Z}=\mathbf{X}_c\mathbf{W}\in\mathbb{R}^{N\times k}`

    Parameters
    ----------
    n_components : int
        主成分数 :math:`k`。
    center : bool, default=True
        True の場合、特徴量平均を引いてからSVDを行う。
    whiten : bool, default=False
        True の場合、transform 出力 :math:`\mathbf{Z}` を
        :math:`\sqrt{\lambda_i}`（成分分散）で割り、各成分を単位分散化する。
    eps : float, default=1e-12
        数値安定化（ゼロ割回避）用の小さい定数。
    dtype : torch.dtype, optional
        fit/transform 時に内部でこのdtypeへ揃える（指定しない場合は入力に従う）。
    device : str | torch.device, optional
        fit/transform 時に内部でこのdeviceへ揃える（指定しない場合は入力に従う）。

    Attributes
    ----------
    mean_ : torch.Tensor
        特徴量平均 :math:`\boldsymbol{\mu}`。形状 (d,)。
        center=False の場合はゼロベクトル。
    loadings_ : torch.Tensor
        loadings（主軸）: :math:`\mathbf{W}`。形状 (d, k)。
        各列は直交規格化される。
    components_ : torch.Tensor
        sklearn互換の別名。loadings_ と同一参照。
    singular_values_ : torch.Tensor
        特異値 :math:`(\sigma_1,\ldots,\sigma_k)`。形状 (k,)。
    explained_variance_ : torch.Tensor
        各成分の分散 :math:`\lambda_i`。形状 (k,)。
        本実装では **分母N** の流儀で :math:`\lambda_i=\sigma_i^2 / N` を採用する。
    explained_variance_ratio_ : torch.Tensor
        保持したk成分内で正規化した寄与率。
        :math:`\lambda_i / \sum_{j=1}^{k}\lambda_j`。形状 (k,)。
    n_samples_ : int
        fit に用いたサンプル数 :math:`N`。
    fitted_ : bool
        学習済みフラグ。

    Notes
    -----
    - SVDは `torch.linalg.svd(Xc, full_matrices=False)` を用いる。
      これは厳密だが、大規模 (N,dが非常に大きい) ではメモリ/計算が重い。
      近似SVD（randomized等）を導入したい場合は別途設計する。
    - explained_variance_ratio_ は「上位k内での正規化」である点に注意。
      全成分（rank全体）でのEVRが必要なら全特異値が必要。

    Raises
    ------
    ValueError
        入力shape不正、n_componentsが範囲外など。
    TypeError
        入力がtorch.Tensorでない、または浮動小数点でない場合。

    Examples
    --------
    >>> import torch
    >>> X = torch.randn(1000, 256, device="cuda")
    >>> pca = PCA(n_components=32, center=True, whiten=False).fit(X)
    >>> Z = pca.transform(X)  # (1000, 32)
    >>> pca.save("pca.pt")
    >>> pca2 = PCA.load("pca.pt", map_location="cuda")
    >>> torch.allclose(pca.loadings, pca2.loadings)
    True
    """

    def __init__(
        self,
        n_components: int,
        *,
        center: bool = True,
        whiten: bool = False,
        eps: float = 1e-12,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(f"n_components は正のintである必要があります: got {n_components}")
        if not isinstance(center, bool):
            raise TypeError("center は bool である必要があります。")
        if not isinstance(whiten, bool):
            raise TypeError("whiten は bool である必要があります。")
        if not isinstance(eps, (float, int)) or float(eps) <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {eps}")

        self.n_components: int = n_components
        self.center: bool = center
        self.whiten: bool = whiten
        self.eps: float = float(eps)

        self._dtype: Optional[torch.dtype] = dtype
        self._device: Optional[torch.device] = torch.device(device) if device is not None else None

        self.mean_: Optional[Tensor] = None
        self.loadings_: Optional[Tensor] = None
        self.components_: Optional[Tensor] = None
        self.singular_values_: Optional[Tensor] = None
        self.explained_variance_: Optional[Tensor] = None
        self.explained_variance_ratio_: Optional[Tensor] = None
        self.n_samples_: Optional[int] = None
        self.fitted_: bool = False

    @property
    def loadings(self) -> Tensor:
        """loadings_（主軸）への別名アクセス。

        Returns
        -------
        loadings : torch.Tensor
            形状 (d, k) の主軸行列。

        Raises
        ------
        RuntimeError
            未学習の場合。
        """
        if not self.fitted_ or self.loadings_ is None:
            raise RuntimeError("PCAは未学習です。fit() を先に呼んでください。")
        return self.loadings_

    @property
    def mean(self) -> Tensor:
        """mean_（特徴量平均）への別名アクセス。"""
        if not self.fitted_ or self.mean_ is None:
            raise RuntimeError("PCAは未学習です。fit() を先に呼んでください。")
        return self.mean_

    def fit(self, X: Tensor) -> "PCA":
        """PCAを学習する（中心化→SVD）。

        Parameters
        ----------
        X : torch.Tensor
            入力データ。形状 (N, d) の浮動小数点テンソル。

        Returns
        -------
        self : PCA
            学習済みインスタンス。

        Raises
        ------
        ValueError
            n_components が min(N, d) を超える場合など。
        TypeError
            入力がtorch.Tensorでない、または浮動小数点でない場合。
        """
        X = _as_tensor_2d(X, dtype=self._dtype, device=self._device)
        n, d = int(X.shape[0]), int(X.shape[1])

        k = self.n_components
        if k > min(n, d):
            raise ValueError(f"n_components={k} は min(N,d)={min(n,d)} 以下である必要があります。")

        if self.center:
            mean = X.mean(dim=0)  # (d,)
            Xc = X - mean
        else:
            mean = torch.zeros(d, dtype=X.dtype, device=X.device)
            Xc = X

        # Xc = U S Vh
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        S_k = S[:k]  # (k,)
        V_k = Vh[:k, :].T.contiguous()  # (d, k)

        explained_var = (S_k * S_k) / float(n)  # lambda_i = sigma_i^2 / N
        evr = _safe_div(explained_var, explained_var.sum(), eps=self.eps)

        self.mean_ = mean
        self.loadings_ = V_k
        self.components_ = self.loadings_
        self.singular_values_ = S_k
        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = evr
        self.n_samples_ = n
        self.fitted_ = True
        return self

    def transform(self, X: Tensor) -> Tensor:
        """学習済みPCA空間へ射影する。

        Parameters
        ----------
        X : torch.Tensor
            入力データ。形状 (N, d)。

        Returns
        -------
        Z : torch.Tensor
            主成分得点。形状 (N, k)。
            whiten=True の場合、各列を sqrt(explained_variance_) で割る。

        Raises
        ------
        RuntimeError
            未学習の場合。
        ValueError
            入力の次元 d が学習時と一致しない場合。
        """
        if not self.fitted_:
            raise RuntimeError("PCAは未学習です。fit() を先に呼んでください。")
        assert self.mean_ is not None
        assert self.loadings_ is not None
        assert self.explained_variance_ is not None

        X = _as_tensor_2d(X, dtype=self.mean_.dtype, device=self.mean_.device)
        if int(X.shape[1]) != int(self.mean_.shape[0]):
            raise ValueError(f"入力d={int(X.shape[1])}が学習時d={int(self.mean_.shape[0])}と一致しません。")

        Xc = X - self.mean_ if self.center else X
        Z = Xc @ self.loadings_

        if self.whiten:
            Z = _safe_div(Z, torch.sqrt(self.explained_variance_), eps=self.eps)
        return Z

    def fit_transform(self, X: Tensor) -> Tensor:
        """学習と変換をまとめて実行する。

        Parameters
        ----------
        X : torch.Tensor
            入力データ。形状 (N, d)。

        Returns
        -------
        Z : torch.Tensor
            主成分得点。形状 (N, k)。
        """
        self.fit(X)
        return self.transform(X)

    def _state_dict(self) -> Dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("未学習のため保存できません。fit() を先に呼んでください。")
        assert self.mean_ is not None
        assert self.loadings_ is not None
        assert self.singular_values_ is not None
        assert self.explained_variance_ is not None
        assert self.explained_variance_ratio_ is not None
        assert self.n_samples_ is not None

        state = PCAState(
            n_components=self.n_components,
            center=self.center,
            whiten=self.whiten,
            eps=self.eps,
            mean=self.mean_.detach().cpu(),
            loadings=self.loadings_.detach().cpu(),
            singular_values=self.singular_values_.detach().cpu(),
            explained_variance=self.explained_variance_.detach().cpu(),
            explained_variance_ratio=self.explained_variance_ratio_.detach().cpu(),
            n_samples=int(self.n_samples_),
            dtype=str(self.mean_.dtype),
            device=str(self.mean_.device),
        )
        return {"pca_state": state}

    def save(self, path: PathLike) -> None:
        """学習済みパラメータを保存する。

        Parameters
        ----------
        path : str | pathlib.Path
            保存先パス（例: "pca.pt"）。

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            未学習の場合。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), p)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        map_location: Optional[Union[str, torch.device]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PCA":
        """保存済みPCAを読み込む。

        Parameters
        ----------
        path : str | pathlib.Path
            save() で保存した .pt へのパス。
        map_location : str | torch.device, optional
            torch.load に渡す map_location（例: "cpu" / "cuda"）。
        device : str | torch.device, optional
            読み込み後に明示的に移すdevice。指定した場合 map_location より優先。
        dtype : torch.dtype, optional
            読み込み後にこのdtypeへキャストする。

        Returns
        -------
        pca : PCA
            復元されたPCAインスタンス。

        Raises
        ------
        FileNotFoundError
            path が存在しない場合。
        ValueError
            チェックポイント形式が不正な場合。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        payload = torch.load(p, map_location=map_location)
        if not isinstance(payload, dict) or "pca_state" not in payload:
            raise ValueError("PCAチェックポイント形式が不正です（'pca_state' がありません）。")

        state = payload["pca_state"]
        if not isinstance(state, PCAState):
            raise ValueError("PCAチェックポイント形式が不正です（pca_state が PCAState ではありません）。")

        pca = cls(
            n_components=state.n_components,
            center=state.center,
            whiten=state.whiten,
            eps=state.eps,
            dtype=dtype,
            device=torch.device(device) if device is not None else None,
        )

        tgt_device: Optional[torch.device]
        if device is not None:
            tgt_device = torch.device(device)
        elif map_location is not None:
            tgt_device = torch.device(map_location)
        else:
            tgt_device = None

        mean = state.mean
        loadings = state.loadings
        sv = state.singular_values
        ev = state.explained_variance
        evr = state.explained_variance_ratio

        if dtype is not None:
            mean = mean.to(dtype=dtype)
            loadings = loadings.to(dtype=dtype)
            sv = sv.to(dtype=dtype)
            ev = ev.to(dtype=dtype)
            evr = evr.to(dtype=dtype)

        if tgt_device is not None:
            mean = mean.to(device=tgt_device)
            loadings = loadings.to(device=tgt_device)
            sv = sv.to(device=tgt_device)
            ev = ev.to(device=tgt_device)
            evr = evr.to(device=tgt_device)

        pca.mean_ = mean
        pca.loadings_ = loadings
        pca.components_ = pca.loadings_
        pca.singular_values_ = sv
        pca.explained_variance_ = ev
        pca.explained_variance_ratio_ = evr
        pca.n_samples_ = int(state.n_samples)
        pca.fitted_ = True
        return pca