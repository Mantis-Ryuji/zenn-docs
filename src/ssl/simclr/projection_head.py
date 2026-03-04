from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ProjectionHeadConfig:
    """SimCLR 風 projection head（2層 MLP + BN）の設定。

    SimCLR / InfoNCE 系では、encoder 出力特徴をそのまま損失に入れず、MLP により別の埋め込み空間へ
    写像してから対照損失を最適化することが多い。本設定は、その 2 層 MLP（中間に BN + ReLU）と
    最終 L2 正規化の有無を制御する。

    Parameters
    ----------
    in_dim : int, default=2048
        入力次元。入力 x の shape は (B, in_dim)。
        通常は ResNet-50 の global average pooling 出力（2048）に対応する。
    hidden_dim : int, default=2048
        中間層次元。
    out_dim : int, default=128
        出力次元。出力 z の shape は (B, out_dim)。
    l2_normalize : bool, default=True
        出力 z を特徴次元（dim=1）で L2 正規化するかどうか。
        True の場合、cosine 類似度（内積）ベースの損失と整合しやすくなる。
    l2_eps : float, default=1e-12
        L2 正規化（F.normalize）の数値安定化項。正値を要求する。
    bn_eps : float, default=1e-5
        BatchNorm1d の eps。正値を要求する。
    bn_momentum : float, default=0.1
        BatchNorm1d の momentum。通常は 0 < momentum < 1 を推奨する。

    Notes
    -----
    - 構成は Linear(bias=False) → BN → ReLU → Linear(bias=True)。
      先頭 Linear を bias=False にして BN にオフセット学習を任せるのは典型的な実装流儀。
    - BatchNorm を用いるため、学習時は十分なバッチサイズ（または勾配蓄積等）が望ましい。
      小バッチ環境では統計が不安定になり得る点に注意。
    """

    in_dim: int = 2048
    hidden_dim: int = 2048
    out_dim: int = 128
    l2_normalize: bool = True
    l2_eps: float = 1e-12
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1


class MLPProjectionHead(nn.Module):
    """2 層 MLP + BatchNorm による projection head。

    Encoder が出力する特徴を、対照損失で用いる埋め込みへ写像するためのヘッド。
    SimCLR では「損失は projection 空間で最適化し、下流タスクでは encoder 特徴を使う」
    という使い分けが典型である。

    Notes
    -----
    - 本実装は ProjectionHeadConfig に従い、Linear → BN → ReLU → Linear を適用する。
    - cfg.l2_normalize=True の場合、出力は特徴次元（dim=1）で L2 正規化される。
    - 本モジュールは分類器ではなく、表現学習用の補助ヘッドである（推論用途では外すことが多い）。
    """

    def __init__(self, cfg: ProjectionHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ProjectionHeadConfig()

        self._validate_cfg(self.cfg)

        self.fc1 = nn.Linear(self.cfg.in_dim, self.cfg.hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.cfg.hidden_dim, eps=self.cfg.bn_eps, momentum=self.cfg.bn_momentum)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim, bias=True)

        self._init_weights()

    @staticmethod
    def _validate_cfg(cfg: ProjectionHeadConfig) -> None:
        if int(cfg.in_dim) < 1:
            raise ValueError(f"in_dim は 1 以上: got {cfg.in_dim}")
        if int(cfg.hidden_dim) < 1:
            raise ValueError(f"hidden_dim は 1 以上: got {cfg.hidden_dim}")
        if int(cfg.out_dim) < 1:
            raise ValueError(f"out_dim は 1 以上: got {cfg.out_dim}")
        if float(cfg.l2_eps) <= 0.0:
            raise ValueError(f"l2_eps は正: got {cfg.l2_eps}")
        if float(cfg.bn_eps) <= 0.0:
            raise ValueError(f"bn_eps は正: got {cfg.bn_eps}")
        if not (0.0 < float(cfg.bn_momentum) < 1.0):
            raise ValueError(f"bn_momentum は (0,1) が推奨: got {cfg.bn_momentum}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: # type: ignore
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """前向き計算（projection）。

        Parameters
        ----------
        x : torch.Tensor
            入力特徴。shape は (B, in_dim)。
            通常は encoder の出力を与える。

        Returns
        -------
        torch.Tensor
            出力埋め込み z。shape は (B, out_dim)。
            cfg.l2_normalize=True の場合、L2 正規化された z を返す。

        Raises
        ------
        TypeError
            x が torch.Tensor でない場合。
        ValueError
            - x が 2 次元 (B, D) でない場合
            - 特徴次元 D が cfg.in_dim と一致しない場合

        Notes
        -----
        - 正規化は F.normalize(z, p=2, dim=1, eps=cfg.l2_eps) を用いる。
        - 本メソッドは dtype/device の変換を行わない（呼び出し側で管理する）。
        """
        if not isinstance(x, Tensor): # type: ignore
            raise TypeError(f"x must be torch.Tensor. got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x は 2 次元 (B, D) が必要です: got shape={tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_dim:
            raise ValueError(f"feature dim mismatch: expected D={self.cfg.in_dim}, got D={x.shape[1]}")

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        z = self.fc2(x)

        if self.cfg.l2_normalize:
            z = F.normalize(z, p=2.0, dim=1, eps=self.cfg.l2_eps)

        return z