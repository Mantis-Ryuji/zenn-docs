from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class NTXentLossConfig:
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss の設定。

    Parameters
    ----------
    temperature : float, default=0.5
        温度パラメータ τ。
    eps : float, default=1e-12
        L2 正規化および数値安定化（log/softmax）用の微小量。
    reduction : str, default="mean"
        "mean" または "sum"。
    check_l2_normalized : bool, default=True
        入力が既に L2 正規化されているかを簡易チェックするかどうか。
    l2_atol : float, default=1e-3
        L2 正規化チェックの許容誤差（||z||2 ≈ 1 の atol）。
    """
    temperature: float = 0.5
    eps: float = 1e-12
    reduction: str = "mean"
    check_l2_normalized: bool = True
    l2_atol: float = 1e-3


class NTXentLoss(nn.Module):
    """NT-Xent loss（SimCLR の contrastive loss）。

    入力は 2-view を連結した埋め込みで、インデックス構造は
    (0,1), (2,3), ..., (2B-2, 2B-1) が正例ペアであることを仮定する。

    Shapes
    ------
    - Input:  z # (2B, 128)
    - Output: loss # ()  (scalar)

    Notes
    -----
    - 内部で一応 L2 正規化を行う（Normalize=True の前提を強制する）。
      さらに `check_l2_normalized=True` の場合は、入力のノルムが 1 に近いかをチェックする。
    - 類似度は cosine similarity（正規化後の内積）を用いる。
    - 自己類似（i==i）は除外する（logits の対角を -inf に落とす）。

    Raises
    ------
    TypeError
        z が torch.Tensor でない場合。
    ValueError
        形状が (2B, D) でない、2B が偶数でない、D が 1 未満、温度が正でない等。
    """

    def __init__(self, cfg: NTXentLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NTXentLossConfig()

        if float(self.cfg.temperature) <= 0.0:
            raise ValueError(f"temperature は正である必要があります: got {self.cfg.temperature}")
        if float(self.cfg.eps) <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {self.cfg.eps}")
        if self.cfg.reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction は 'mean' または 'sum': got {self.cfg.reduction}")
        if float(self.cfg.l2_atol) <= 0.0:
            raise ValueError(f"l2_atol は正である必要があります: got {self.cfg.l2_atol}")

    def forward(self, z: Tensor) -> Tensor:
        """損失を計算する。

        Parameters
        ----------
        z : torch.Tensor
            2-view を連結した埋め込み。z # (2B, 128)

        Returns
        -------
        torch.Tensor
            損失スカラー。loss # ()

        Raises
        ------
        TypeError
            z が torch.Tensor でない場合。
        ValueError
            z の次元・shape が不正な場合。
        """
        if not isinstance(z, Tensor): # type: ignore
            raise TypeError(f"z must be torch.Tensor. got {type(z)}")
        if z.ndim != 2:
            raise ValueError(f"z は 2 次元 (2B, D) である必要があります: got shape={tuple(z.shape)}")

        n, d = z.shape  # n # 2B, d # D(=128)
        if n < 2 or (n % 2) != 0:
            raise ValueError(f"先頭次元は偶数 (2B) である必要があります: got {n}")
        if d < 1:
            raise ValueError(f"埋め込み次元 D は 1 以上: got {d}")

        # 事前条件：入力は（基本）L2 正規化されている想定だが、内部でも一応正規化する
        if self.cfg.check_l2_normalized:
            norms_in = z.norm(p=2, dim=1)  # norms_in # (2B,)
            max_dev = (norms_in - 1.0).abs().max().item()
            if max_dev > float(self.cfg.l2_atol):
                raise ValueError(
                    "入力 z は L2 正規化されている前提（||z||2≈1）です。"
                    f" max | ||z||2 - 1 | = {max_dev:.3e} > atol={self.cfg.l2_atol:.3e}"
                )

        z = F.normalize(z, p=2.0, dim=1, eps=self.cfg.eps)  # z # (2B, D)

        # cosine similarity logits: (2B,2B)
        logits = (z @ z.t()) / float(self.cfg.temperature)  # logits # (2B, 2B)

        # self-similarity mask: diag -> -inf
        diag = torch.eye(n, device=logits.device, dtype=torch.bool)  # diag # (2B,2B)
        logits = logits.masked_fill(diag, float("-inf"))

        # pos(i): 0<->1, 2<->3, ...
        idx = torch.arange(n, device=z.device)  # idx # (2B,)
        pos = idx ^ 1  # pos # (2B,)

        # cross entropy over candidates (excluding self via -inf)
        loss_vec = F.cross_entropy(logits, pos, reduction="none")  # loss_vec # (2B,)

        if self.cfg.reduction == "mean":
            return loss_vec.mean()
        return loss_vec.sum()