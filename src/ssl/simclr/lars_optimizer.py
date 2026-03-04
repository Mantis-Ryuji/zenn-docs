from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"\.bias$",
    r"\.(?:bn\d*|norm1|downsample\.1|bn1)\.(?:weight|bias)$",
)


@dataclass(frozen=True)
class LARSConfig:
    """LARS（Layer-wise Adaptive Rate Scaling）最適化の設定（SimCLR 想定）。

    SimCLR 論文で使われる「大規模バッチ学習向けの SGD 系最適化」として知られる LARS のうち、
    本実装が採用する最小構成のハイパーパラメータを保持する。

    本実装の特徴は以下。
    - weight decay は decoupled（AdamW 型）ではなく **coupled**（勾配に加算）で適用する
    - layer-wise adaptation（trust ratio）をパラメータ名の正規表現で除外できる
    - 除外デフォルトは bias と BN/Norm 系（DEFAULT_EXCLUDE_PATTERNS）

    Parameters
    ----------
    lr : float
        ベース学習率（各 param group の既定 lr）。正値を要求する。
        実際の更新量は trust ratio によりスケールされた学習率（scaled_lr = lr * trust）で決まる。
    momentum : float, default=0.9
        モメンタム係数。0 <= momentum < 1 を要求する。
    use_nesterov : bool, default=False
        True の場合、Nesterov momentum を用いる。
    weight_decay : float, default=1e-6
        L2 weight decay 係数（coupled: grad += wd * w）。0 以上を要求する。
    exclude_from_weight_decay : tuple[str, ...] | None, default=None
        weight decay を適用しないパラメータ名の正規表現パターン（re.search で判定）。
        None の場合はデフォルト（bias と BN/Norm 系）を除外する。
    exclude_from_layer_adaptation : tuple[str, ...] | None, default=None
        trust ratio（layer-wise adaptation）を適用しないパラメータ名の正規表現パターン。
        None の場合は exclude_from_weight_decay と同じ扱い（典型的な流儀）。
    eeta : float, default=0.001
        trust ratio のスケール係数。正値を要求する。
        値が大きいほど layer-wise スケーリングが強くなる。
    eps : float, default=0.0
        trust ratio の分母に加える数値安定化項。0 以上を要求する。
        勾配ノルムが極小のときの過大な trust ratio を抑制する用途で使う。

    Raises
    ------
    ValueError
        lr <= 0、momentum が [0,1) 外、weight_decay < 0、eeta <= 0、eps < 0 の場合。

    Notes
    -----
    - 典型的には bias と BN/Norm 系は weight decay / layer adaptation の両方から除外する。
      これは統計量（running mean/var 等）やバイアス項に対するスケーリングが不安定化しやすいため。
    - 本設定は「名前ベースでの除外」を行うため、optimizer には named_parameters() を渡す前提となる。
    """

    lr: float
    momentum: float = 0.9
    use_nesterov: bool = False
    weight_decay: float = 1e-6
    exclude_from_weight_decay: tuple[str, ...] | None = None
    exclude_from_layer_adaptation: tuple[str, ...] | None = None
    eeta: float = 0.001
    eps: float = 0.0


class LARS(Optimizer):
    """SimCLR 想定の LARS Optimizer（named_parameters 前提）。

    `named_parameters()` 等で得られる (name, parameter) 列を受け取り、
    パラメータ名に基づいて以下を制御する。
    - coupled weight decay の適用/除外
    - layer-wise adaptation（trust ratio）の適用/除外

    Parameters
    ----------
    named_params : Iterable[tuple[str, torch.nn.Parameter]]
        (name, parameter) の列。空は不可。
        name は除外判定（正規表現）に用いられる。
    cfg : LARSConfig
        最適化の設定。

    Raises
    ------
    ValueError
        named_params が空の場合、または cfg が不正な場合。
    RuntimeError
        step() で sparse gradient が検出された場合（非対応）。

    Notes
    -----
    - lr は param group の "lr" が存在すればそれを優先し、無ければ cfg.lr を用いる。
    - `self._name_by_param` は id(parameter) → name の対応を保持し、step() で参照する。
      これにより、param group を経由しても名前情報を復元できる。
    - 本実装は PyTorch Optimizer の一般的な慣習どおり、loss などを返さず None を返す。
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]],
        cfg: LARSConfig,
    ) -> None:
        self._validate_cfg(cfg)
        self.cfg = cfg

        named = list(named_params)
        if len(named) == 0:
            raise ValueError("named_params is empty.")

        exclude_wd = cfg.exclude_from_weight_decay if cfg.exclude_from_weight_decay is not None else DEFAULT_EXCLUDE_PATTERNS
        exclude_la = cfg.exclude_from_layer_adaptation if cfg.exclude_from_layer_adaptation is not None else exclude_wd

        self._re_wd = [re.compile(p) for p in exclude_wd]
        self._re_la = [re.compile(p) for p in exclude_la]

        params = [p for _, p in named]
        super().__init__(params, defaults={"lr": float(cfg.lr)})

        self._name_by_param: dict[int, str] = {id(p): n for n, p in named}

    @staticmethod
    def _validate_cfg(cfg: LARSConfig) -> None:
        if float(cfg.lr) <= 0.0:
            raise ValueError(f"lr must be positive. got {cfg.lr}")
        if not (0.0 <= float(cfg.momentum) < 1.0):
            raise ValueError(f"momentum must be in [0,1). got {cfg.momentum}")
        if float(cfg.weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be >=0. got {cfg.weight_decay}")
        if float(cfg.eeta) <= 0.0:
            raise ValueError(f"eeta must be positive. got {cfg.eeta}")
        if float(cfg.eps) < 0.0:
            raise ValueError(f"eps must be >=0. got {cfg.eps}")

    @staticmethod
    def _match_any(patterns: list[re.Pattern[str]], name: str) -> bool:
        return any(p.search(name) is not None for p in patterns)

    def _use_weight_decay(self, name: str) -> bool:
        if float(self.cfg.weight_decay) == 0.0:
            return False
        return not self._match_any(self._re_wd, name)

    def _do_layer_adaptation(self, name: str) -> bool:
        return not self._match_any(self._re_la, name)

    def step(self) -> None:  # type: ignore[override]
        """1 step 分のパラメータ更新を行う。

        各パラメータ p について、以下の流れで更新する。
        1) 必要なら coupled weight decay を勾配に加算する（grad += wd * p）
        2) 必要なら trust ratio を計算して学習率をスケールする（scaled_lr = lr * trust）
        3) momentum（および必要なら Nesterov）で更新量を構成し、p を更新する
        4) momentum buffer を state に保存する

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            sparse gradient が存在する場合（LARS は非対応）。

        Notes
        -----
        - trust ratio はパラメータノルムと勾配ノルムから計算する。
        ただし、除外パターンに一致する場合は trust=1 として layer adaptation をスキップする。
        - trust ratio 計算は float 変換してノルムを計算する（dtype の影響を受けにくくするため）。
        mixed precision（fp16/bf16）環境でも極端な underflow/overflow を避ける意図。
        - 勾配が None のパラメータはスキップする（標準的な最適化ループ挙動）。
        - weight_decay=0 の場合、weight decay の分岐は常に無効化される。
        """
        lr0 = float(self.cfg.lr)
        m = float(self.cfg.momentum)
        wd = float(self.cfg.weight_decay)
        eeta = float(self.cfg.eeta)
        eps = float(self.cfg.eps)

        with torch.no_grad():
            for group in self.param_groups:
                lr = float(group.get("lr", lr0))

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("LARS does not support sparse gradients.")

                    name = self._name_by_param.get(id(p), "")
                    g = p.grad.detach()

                    # coupled weight decay
                    if self._use_weight_decay(name):
                        g = g.add(p, alpha=wd)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    v: Tensor = state["momentum_buffer"]

                    trust = 1.0
                    if self._do_layer_adaptation(name):
                        w_norm = torch.norm(p.detach().float(), p=2)
                        g_norm = torch.norm(g.detach().float(), p=2)
                        if (w_norm > 0).item() and (g_norm > 0).item():
                            trust = float((eeta * w_norm / (g_norm + eps)).item())

                    scaled_lr = lr * trust

                    # v_next = m*v + scaled_lr * g
                    next_v = v.mul(m).add(g, alpha=scaled_lr)

                    if self.cfg.use_nesterov:
                        update = next_v.mul(m).add(g, alpha=scaled_lr)
                    else:
                        update = next_v

                    p.add_(update, alpha=-1.0)
                    v.copy_(next_v)