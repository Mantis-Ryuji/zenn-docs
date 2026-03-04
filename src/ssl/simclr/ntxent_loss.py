from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Reduction: TypeAlias = Literal["mean", "sum"]


@dataclass(frozen=True)
class NTXentLossConfig:
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss の設定。

    SimCLR で用いられる対照損失（NT-Xent）のハイパーパラメータを保持する。
    本実装では入力埋め込み z を内部で L2 正規化し、正規化後の内積（cosine 類似度）を
    温度でスケーリングした logits に対して cross entropy を計算する。

    Parameters
    ----------
    temperature : float, default=0.1
        温度パラメータ。logits のスケールを制御する。
        小さいほど分布が鋭くなり、学習が不安定になり得るため、正値を要求する。
    l2_eps : float, default=1e-12
        L2 正規化（F.normalize）の数値安定化項。正値を要求する。
    reduction : {"mean", "sum"}, default="mean"
        損失の集約方法。
        - "mean": 1 サンプル（アンカー）あたりの平均損失
        - "sum":  全アンカーの損失和

    Notes
    -----
    - temperature は学習挙動に強く影響する（特にバッチサイズや埋め込み次元に依存）。
      変更する場合は学習率やバッチサイズとのバランスも合わせて調整すること。
    """

    temperature: float = 0.1
    l2_eps: float = 1e-12
    reduction: Reduction = "mean"


class NTXentLoss(nn.Module):
    """NT-Xent loss（SimCLR の対照損失）。

    入力は 2-view を連結した埋め込みで、インデックス構造として
    (0,1), (2,3), ..., (2B-2, 2B-1) が正例ペアであることを仮定する。
    すなわち、各 i の正例インデックスは `i ^ 1`（0<->1, 2<->3, ...）で与えられる。

    Notes
    -----
    - 入力 z は内部で常に L2 正規化される（外部での正規化は必須ではない）。
    - 類似度は正規化後の内積（cosine 類似度）を用い、temperature でスケーリングする。
    - 自己類似（i==i）は候補から除外するため、logits の対角成分を -inf に置換する。
      これにより cross entropy の分母から自己項が除外される。
    - 本実装は「各アンカー i が、候補集合（自分以外の全サンプル）から正例を当てる分類問題」
      として損失を定義する（対称性のため 2B 個すべてをアンカーとして平均/和を取る）。

    Raises
    ------
    TypeError
        z が torch.Tensor でない場合。
    ValueError
        - z が 2 次元でない場合
        - 先頭次元が偶数でない場合（2B を要求）
        - 埋め込み次元が 1 未満の場合
        - temperature / l2_eps / reduction が不正な場合
    """

    def __init__(self, cfg: NTXentLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NTXentLossConfig()
        self._validate_cfg(self.cfg)

    @staticmethod
    def _validate_cfg(cfg: NTXentLossConfig) -> None:
        if float(cfg.temperature) <= 0.0:
            raise ValueError(f"temperature は正である必要があります: got {cfg.temperature}")
        if float(cfg.l2_eps) <= 0.0:
            raise ValueError(f"l2_eps は正である必要があります: got {cfg.l2_eps}")
        if cfg.reduction not in ("mean", "sum"):
            raise ValueError(f"reduction は 'mean' または 'sum': got {cfg.reduction}")

    def forward(self, z: Tensor) -> Tensor:
        """NT-Xent 損失を計算する。

        Parameters
        ----------
        z : torch.Tensor
            2-view を連結した埋め込み。shape は (2B, D)。
            正例ペアは (0,1), (2,3), ... の順に並んでいることを前提とする。

        Returns
        -------
        torch.Tensor
            損失スカラー。
            cfg.reduction="mean" の場合は全アンカー平均、"sum" の場合は全アンカー和。

        Raises
        ------
        TypeError
            z が torch.Tensor でない場合。
        ValueError
            - z が 2 次元 (2B, D) でない場合
            - 先頭次元が偶数でない場合
            - 埋め込み次元 D が 1 以上でない場合

        Notes
        -----
        - z は内部で F.normalize(..., dim=1, eps=cfg.l2_eps) により L2 正規化される。
        - logits は (z @ z.T) / temperature として計算する（自己項は -inf で除外）。
        - 正例ラベルは `pos = arange(2B) ^ 1` により構成する。
        - dtype/device の変換は行わない（呼び出し側で管理する）。
        """
        if not isinstance(z, Tensor): # type: ignore
            raise TypeError(f"z must be torch.Tensor. got {type(z).__name__}")
        if z.ndim != 2:
            raise ValueError(f"z は 2 次元 (2B, D) が必要です: got shape={tuple(z.shape)}")

        n, d = z.shape  # n == 2B
        if n < 2 or (n % 2) != 0:
            raise ValueError(f"先頭次元は偶数 (2B) である必要があります: got {n}")
        if d < 1:
            raise ValueError(f"埋め込み次元 D は 1 以上: got {d}")

        # normalize: (2B, D)
        z = F.normalize(z, p=2.0, dim=1, eps=float(self.cfg.l2_eps))

        # cosine similarity logits: (2B, 2B)
        logits = (z @ z.t()) / float(self.cfg.temperature)

        # exclude self-similarity
        diag = torch.eye(n, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diag, float("-inf"))

        # pos(i): 0<->1, 2<->3, ...
        idx = torch.arange(n, device=logits.device)
        pos = idx ^ 1

        # cross entropy over candidates (self is -inf)
        loss_vec = F.cross_entropy(logits, pos, reduction="none")

        if self.cfg.reduction == "mean":
            return loss_vec.mean()
        return loss_vec.sum()