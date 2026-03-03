from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class ProjectionHeadConfig:
    """SimCLR 風 projection head（2層 MLP）の設定。

    Parameters
    ----------
    in_dim : int, default=2048
        入力次元。入力 x の形状は x # (B, in_dim)。
    hidden_dim : int, default=2048
        中間層次元。SimCLR では in_dim と同じにする設定がよく使われる。
    out_dim : int, default=128
        出力次元。出力 z の形状は z # (B, out_dim)。
    l2_normalize : bool, default=True
        出力 z を L2 正規化（最後次元）するかどうか。
    eps : float, default=1e-12
        L2 正規化の数値安定化項。
    """
    in_dim: int = 2048
    hidden_dim: int = 2048
    out_dim: int = 128
    l2_normalize: bool = True
    eps: float = 1e-12


class MLPProjectionHead(nn.Module):
    """2層 MLP による projection head。

    SimCLR の典型形：
    - Linear(in_dim -> hidden_dim)
    - ReLU
    - Linear(hidden_dim -> out_dim)
    - (optional) L2 normalize

    Shapes
    ------
    - Input:  x # (B, 2048)
    - Output: z # (B, 128)   ※ l2_normalize=True の場合は ||z||_2 = 1

    Notes
    -----
    - 本クラスは「backbone 出力 h # (B, 2048)」を「損失空間 z # (B, 128)」へ写像する用途を想定する。
    - LayerNorm / BatchNorm を入れたい場合は、SimCLR の実装流儀に合わせて別途差し込む。
    """

    def __init__(self, cfg: ProjectionHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ProjectionHeadConfig()

        if self.cfg.in_dim < 1:
            raise ValueError(f"in_dim は 1 以上: got {self.cfg.in_dim}")
        if self.cfg.hidden_dim < 1:
            raise ValueError(f"hidden_dim は 1 以上: got {self.cfg.hidden_dim}")
        if self.cfg.out_dim < 1:
            raise ValueError(f"out_dim は 1 以上: got {self.cfg.out_dim}")
        if self.cfg.eps <= 0.0:
            raise ValueError(f"eps は正: got {self.cfg.eps}")

        self.fc1 = nn.Linear(self.cfg.in_dim, self.cfg.hidden_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.cfg.hidden_dim, self.cfg.out_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Parameters
        ----------
        x : torch.Tensor
            入力特徴。x # (B, in_dim)

        Returns
        -------
        torch.Tensor
            出力埋め込み。z # (B, out_dim)

        Raises
        ------
        TypeError
            x が torch.Tensor でない場合。
        ValueError
            x が 2 次元でない場合、または in_dim と一致しない場合。
        """
        if not isinstance(x, Tensor): # type: ignore
            raise TypeError(f"x must be torch.Tensor. got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"x は 2 次元 (B, D) である必要があります: got shape={tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_dim:
            raise ValueError(f"feature dim mismatch: expected D={self.cfg.in_dim}, got D={x.shape[1]}")

        x = self.fc1(x)  # x # (B, hidden_dim)
        x = self.act(x)
        z = self.fc2(x)  # z # (B, out_dim)

        if self.cfg.l2_normalize:
            z = torch.nn.functional.normalize(z, p=2.0, dim=1, eps=self.cfg.eps)  # z # (B, out_dim)

        return z