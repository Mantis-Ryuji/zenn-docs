from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"\.bias$",
    r"\.ln\.(?:weight|bias)$",        # LayerNorm2d 内部
    r"\.out_norm\.(?:weight|bias)$",  # 最終 LayerNorm
)
EETA_DEFAULT: float = 0.001

@dataclass(frozen=True)
class LARSConfig:
    """LARS（Layer-wise Adaptive Rate Scaling）の設定。

    Parameters
    ----------
    lr : float
        学習率（ベース学習率）。
    momentum : float, default=0.9
        モメンタム係数。
    use_nesterov : bool, default=False
        Nesterov momentum を使うかどうか。
    weight_decay : float, default=1e-6
        L2 weight decay 係数（**coupled**: grad に加算する）。
        注意：AdamW のような decoupled weight decay ではない。
    exclude_from_weight_decay : tuple[str, ...] | None, default=None
        パラメータ名に対する正規表現のリスト。
        いずれかがマッチした場合、そのパラメータには weight decay を適用しない。
        None の場合、bias と LayerNorm 系をデフォルトで除外する。
    exclude_from_layer_adaptation : tuple[str, ...] | None, default=None
        パラメータ名に対する正規表現のリスト。
        いずれかがマッチした場合、そのパラメータでは trust ratio（layer adaptation）を計算しない。
        None の場合、exclude_from_weight_decay と同じ挙動にする（SimCLR TF 実装の流儀）。
    classic_momentum : bool, default=True
        SimCLR(TF) 実装の classic/popular momentum 分岐。
        - True: v = m*v + (lr*trust)*g, param -= (nesterov? m*v_next + (lr*trust)*g : v_next)
        - False: v = m*v + g, update=(nesterov? m*v_next + g : v_next), param -= (lr*trust)*update
    eeta : float, default=0.001
        trust ratio のスケール係数（論文・TF実装の eeta）。
    eps : float, default=0.0
        ノルムが 0 近傍のときの数値安定化項。
        TF版は where 分岐で 0 除算を避けるため、基本 0.0 で良い。

    Raises
    ------
    ValueError
        lr <= 0、momentum が [0,1) 外、weight_decay < 0、eeta <= 0、eps < 0 の場合。
    """

    lr: float
    momentum: float = 0.9
    use_nesterov: bool = False
    weight_decay: float = 1e-6
    exclude_from_weight_decay: tuple[str, ...] | None = DEFAULT_EXCLUDE_PATTERNS
    exclude_from_layer_adaptation: tuple[str, ...] | None = DEFAULT_EXCLUDE_PATTERNS
    classic_momentum: bool = True
    eeta: float = EETA_DEFAULT
    eps: float = 0.0


class LARS(Optimizer):
    """SimCLR(TF) 実装に合わせた LARS Optimizer（PyTorch）。

    Notes
    -----
    - weight decay は decoupled ではなく coupled（grad += wd * param）。
    - trust ratio:
        - classic_momentum=True:  trust = eeta * ||w|| / ||g||
        - classic_momentum=False: trust = eeta * ||w|| / ||update||
      ただし ||w||==0 または分母が 0 の場合は trust=1 とする（TF版の where と同等）。
    - exclude_* はパラメータ名に対する正規表現検索（re.search）で判定する。
    - デフォルトでは bias と LayerNorm 系を weight decay / layer adaptation から除外する。
      （「最適化（更新）そのもの」は行う。）
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]],
        cfg: LARSConfig,
    ) -> None:
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

        self.cfg = cfg
        self._named_params = list(named_params)
        if len(self._named_params) == 0:
            raise ValueError("named_params is empty.")

        # --- default excludes ---
        exclude_wd = cfg.exclude_from_weight_decay or DEFAULT_EXCLUDE_PATTERNS
        # TF流儀：layer adaptation の除外が None なら weight decay の除外と同じ
        exclude_la = cfg.exclude_from_layer_adaptation if cfg.exclude_from_layer_adaptation is not None else exclude_wd

        self._re_wd: list[re.Pattern[str]] = [re.compile(p) for p in exclude_wd]
        self._re_la: list[re.Pattern[str]] = [re.compile(p) for p in exclude_la]

        params = [p for _, p in self._named_params]
        super().__init__(params, defaults={"lr": cfg.lr})

        self._name_by_param: dict[int, str] = {id(p): n for n, p in self._named_params}

    def _match_any(self, patterns: list[re.Pattern[str]], name: str) -> bool:
        return any(p.search(name) is not None for p in patterns)

    def _use_weight_decay(self, name: str) -> bool:
        """param_name に weight decay を適用するか。"""
        if self.cfg.weight_decay == 0.0:
            return False
        if self._match_any(self._re_wd, name):
            return False
        return True

    def _do_layer_adaptation(self, name: str) -> bool:
        """param_name に layer adaptation（trust ratio）を適用するか。"""
        if self._match_any(self._re_la, name):
            return False
        return True

    def step(self, closure: Callable[[], float] | None = None) -> None:  # type: ignore[override]
        """1 step 更新する。

        Parameters
        ----------
        closure : Callable[[], float] | None
            損失を再計算するための関数。通常は None。
            None でない場合、内部で ``torch.enable_grad()`` 下で呼び出す。
            （型スタブ互換のため float を返す想定）

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            sparse grad が渡された場合（未対応）。
        """
        if closure is not None:
            with torch.enable_grad():
                _ = float(closure())

        with torch.no_grad():
            lr = float(self.cfg.lr)
            m = float(self.cfg.momentum)
            wd = float(self.cfg.weight_decay)
            eeta = float(self.cfg.eeta)
            eps = float(self.cfg.eps)

            for group in self.param_groups:
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
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    v: Tensor = state["momentum_buffer"]

                    if self.cfg.classic_momentum:
                        trust = 1.0
                        if self._do_layer_adaptation(name):
                            w_norm = torch.norm(p, p=2)
                            g_norm = torch.norm(g, p=2)
                            if (w_norm > 0).item() and (g_norm > 0).item():
                                trust = (eeta * w_norm / (g_norm + eps)).item()

                        scaled_lr = lr * trust
                        next_v = v.mul(m).add(g, alpha=scaled_lr)
                        update = next_v.mul(m).add(g, alpha=scaled_lr) if self.cfg.use_nesterov else next_v

                        p.add_(update, alpha=-1.0)
                        v.copy_(next_v)

                    else:
                        next_v = v.mul(m).add(g)
                        update = next_v.mul(m).add(g) if self.cfg.use_nesterov else next_v

                        trust = 1.0
                        if self._do_layer_adaptation(name):
                            w_norm = torch.norm(p, p=2)
                            u_norm = torch.norm(update, p=2)
                            if (w_norm > 0).item() and (u_norm > 0).item():
                                trust = (eeta * w_norm / (u_norm + eps)).item()

                        scaled_lr = lr * trust
                        p.add_(update, alpha=-scaled_lr)
                        v.copy_(next_v)