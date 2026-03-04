from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch
from data_augmentation import SimCLRAugConfig, preprocess_for_train_batch
from lars_optimizer import LARS, LARSConfig
from ntxent_loss import NTXentLoss, NTXentLossConfig
from projection_head import MLPProjectionHead, ProjectionHeadConfig
from resnet_encoder import ResNet50Encoder, ResNet50EncoderConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader

AmpDType = Literal["bf16", "fp16", "none"]


class FitHistory(TypedDict):
    """fit() が返す学習ログ（epoch 単位）。

    Trainer.fit() が返す辞書のスキーマを定義する。ログは「表示・再開・可視化」の用途を想定し、
    1 epoch ごとに 1 要素を append していく。

    Notes
    -----
    - 1 epoch ごとに各キーの list に 1 要素を append する（全キーで同じ長さになる）。
    - loss は 1 epoch 内の「ミニバッチ損失（勾配蓄積で割る前の loss）」の平均。
    - lr は epoch 終了時点の scheduler が保持する LR（先頭 param group）。
    - pos_sim / neg_sim は、各 accumulation window の「最後のミニバッチ」で計算した値を
      epoch 内で平均したもの（厳密に全ミニバッチ平均ではない点に注意）。
    """

    epoch: list[int]
    loss: list[float]
    lr: list[float]
    pos_sim: list[float]
    neg_sim: list[float]


@dataclass(frozen=True)
class SimCLRTrainerConfig:
    """SimCLR 学習の Trainer 設定。

    単一プロセスでの SimCLR 事前学習を想定した設定をまとめる。
    勾配蓄積（accum_steps）により「実効バッチサイズ」を増やし、LARS + warmup + cosine による
    大規模バッチ学習の流儀を模した学習ループを構成する。

    Notes
    -----
    - 学習率スケーリング：
      base_lr は target_global_batch（基準 global batch）に対応する学習率として扱い、
      実際の実効バッチ（batch_size * accum_steps）に対して線形スケールする。
    - スケジューラ：
      warmup_epochs までは線形 warmup、その後は cosine annealing（restart なし）。
      いずれも「optimizer step 単位（勾配蓄積が完了したタイミング）」で更新する。
    - amp_dtype：
      "bf16"/"fp16" は CUDA 環境でのみ有効化され、CUDA 以外では無効化される実装を想定する。
    - チェックポイント：
      last.pt を常に更新し、save_every 間隔で epoch_XXXX.pt を追加保存する。
      keep_last_k により epoch_*.pt の保持数を制限する（last.pt は別扱い）。
    """

    # train loop
    epochs: int = 100
    warmup_epochs: int = 10
    seed: int = 42

    # batch / accumulation
    batch_size: int = 256
    accum_steps: int = 16  # effective batch = batch_size * accum_steps (single-process)
    target_global_batch: int = 4096  # reference batch for base_lr

    # optimizer (LARS)
    base_lr: float = 4.8
    weight_decay: float = 1e-6
    momentum: float = 0.9
    use_nesterov: bool = False

    # loss
    temperature: float = 0.1

    # AMP
    amp_dtype: AmpDType = "bf16"

    # misc
    out_dir: Path = Path("runs/simclr")
    save_every: int = 5  # epochs
    keep_last_k: int = 1  # checkpoints

    # resume
    resume_from: str | Path | Literal["auto", "none"] = "auto"  # checkpoints


class SimCLRModel(nn.Module):
    """Encoder + ProjectionHead からなる SimCLR モデル。

    画像を encoder で特徴抽出し、projection head（2 層 MLP + BN）で対照損失用の埋め込みへ写像する。
    forward() は「埋め込み z のみ」を返し、損失計算（2-view 連結など）は Trainer 側で扱う。

    Parameters
    ----------
    encoder_cfg : ResNet50EncoderConfig | None, optional
        Encoder（ResNet-50）設定。None の場合はデフォルト設定を用いる。
    proj_cfg : ProjectionHeadConfig | None, optional
        Projection head 設定。None の場合はデフォルト設定を用いる。

    Notes
    -----
    - Encoder の出力次元は 2048 を想定し、projection head の in_dim デフォルトも 2048。
    - Projection head はデフォルトで出力を L2 正規化する設定を想定する
      （ただし損失側でも正規化する実装が一般的であり、二重正規化になる可能性がある点に注意）。
    """

    def __init__(
        self,
        *,
        encoder_cfg: ResNet50EncoderConfig | None = None,
        proj_cfg: ProjectionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(encoder_cfg)
        self.projector = MLPProjectionHead(proj_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """前向き計算（画像 → 埋め込み）。

        Parameters
        ----------
        x : torch.Tensor
            入力画像。shape は (B, C, H, W)。
            前処理（正規化・拡張）は呼び出し側（Trainer）で実行する想定。

        Returns
        -------
        torch.Tensor
            埋め込み z。shape は (B, out_dim)（out_dim は projection head 設定による）。

        Notes
        -----
        - 本メソッドは損失計算に必要な 2-view の整形や連結は行わない。
        - dtype/device の変換は行わない（呼び出し側で管理する）。
        """
        h = self.encoder(x)  # (B, 2048)
        z = self.projector(h)  # (B, 128) (l2-normalized by default)
        return z


def _torch_load_checkpoint(path: Path) -> dict[str, object]:
    try:
        obj = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("checkpoint payload must be a dict")
    return obj


def _atomic_write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_json_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"json must be an object/dict, got {type(obj).__name__}.")
    return obj


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_ctx(device: torch.device, amp_dtype: AmpDType):
    if amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown amp_dtype: {amp_dtype}")


class WarmupCosineSchedule:
    """線形 warmup → cosine annealing（restart なし）の step-wise scheduler。

    Optimizer の param group の "lr" を、step() 呼び出しごとに更新するシンプルなスケジューラ。
    Trainer 側で「optimizer step と同じ頻度」で step() を呼ぶことを想定する
    （勾配蓄積を使う場合、ミニバッチごとではなく accumulation 完了ごと）。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        学習率を更新する対象の optimizer。
    warmup_steps : int
        線形 warmup を行う step 数。0 以上かつ total_steps 未満を要求する。
    total_steps : int
        全 step 数（warmup を含む）。正値を要求する。
    base_lrs : list[float]
        各 param group に対応する基準学習率（スケール前の値）。
        optimizer.param_groups と同じ長さである必要がある。

    Notes
    -----
    - step_idx は 0 始まりで内部に保持し、step() のたびに 1 ずつ増える。
    - state_dict / load_state_dict により、学習再開時に同一の lr 推移を復元できる。
    - lr プロパティは先頭 param group の現在 lr を返す（ログ用途）。
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        base_lrs: list[float],
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >=0. got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be >0. got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps must be < total_steps. got warmup_steps={warmup_steps}, total_steps={total_steps}"
            )
        if len(base_lrs) != len(optimizer.param_groups):
            raise ValueError("base_lrs length must match optimizer.param_groups")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = base_lrs
        self.step_idx = 0

    def step(self) -> None:
        """学習率を 1 step 分更新する。

        Notes
        -----
        - warmup 区間では線形に lr を増加させる。
        - warmup 後は cosine annealing（restart なし）で lr を減衰させる。
        - 本メソッドは optimizer.step() を呼ばない（学習率の更新のみ行う）。
        """
        s = self.step_idx
        if s < self.warmup_steps:
            scale = float(s + 1) / float(self.warmup_steps)
        else:
            t = float(s - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            g["lr"] = float(base_lr) * float(scale)

        self.step_idx += 1

    def state_dict(self) -> dict[str, object]:
        """スケジューラ状態を辞書として返す。

        Returns
        -------
        dict[str, object]
            warmup_steps / total_steps / base_lrs / step_idx を含む状態辞書。

        Notes
        -----
        - JSON 化ではなく、PyTorch のチェックポイント（torch.save）に載せる用途を想定する。
        """
        return {
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps),
            "base_lrs": [float(x) for x in self.base_lrs],
            "step_idx": int(self.step_idx),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """state_dict() で保存した状態を復元する。

        Parameters
        ----------
        state : dict[str, object]
            state_dict() が返す形式の辞書。

        Raises
        ------
        ValueError
            状態が不正な場合（step 数の整合性が取れない等）。
        ValueError
            base_lrs の長さが optimizer.param_groups と一致しない場合。

        Notes
        -----
        - 復元後、現在の step_idx に対応する lr を再計算して optimizer に反映する。
        （load_state_dict 自体は step_idx をインクリメントしない）
        """
        warmup_steps = int(state["warmup_steps"])  # type: ignore[arg-type]
        total_steps = int(state["total_steps"])  # type: ignore[arg-type]
        base_lrs = list(state["base_lrs"])  # type: ignore[arg-type]
        step_idx = int(state["step_idx"])  # type: ignore[arg-type]

        if warmup_steps < 0 or total_steps <= 0 or warmup_steps >= total_steps:
            raise ValueError("Invalid scheduler state.")
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler state base_lrs length mismatch.")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [float(x) for x in base_lrs]
        self.step_idx = step_idx

        # restore lr consistent with current step (do not increment step_idx)
        s = self.step_idx
        if s <= 0:
            scale = 0.0
        elif s <= self.warmup_steps:
            scale = float(s) / float(self.warmup_steps)
        else:
            t = float((s - 1) - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            g["lr"] = float(base_lr) * float(scale)

    @property
    def lr(self) -> float:
        """現在の学習率（先頭 param group）。

        Returns
        -------
        float
            optimizer.param_groups[0]["lr"]。
        """
        return float(self.optimizer.param_groups[0]["lr"])


class SimCLRTrainer:
    """SimCLRTrainer（単一プロセス + 勾配蓄積 + チェックポイント）。

    DataLoader から画像バッチを受け取り、2 回の augmentation により 2-view を生成して
    SimCLR の対照学習を行う Trainer。

    主な仕様：
    - 勾配蓄積（accum_steps）により実効バッチを増やす
    - LARS optimizer を使用（named_parameters による除外判定を含む）
    - warmup + cosine の step-wise スケジューラ（optimizer step 単位）
    - last.pt と epoch_*.pt のチェックポイント、training_history.json の保存・再開

    Parameters
    ----------
    cfg : SimCLRTrainerConfig | None, optional
        Trainer 設定。None の場合はデフォルト設定を用いる。
    aug_cfg : SimCLRAugConfig | None, optional
        Augmentation 設定。None の場合はデフォルト設定を用いる。
    model : SimCLRModel | None, optional
        学習するモデル。None の場合はデフォルトの SimCLRModel を生成する。
    device : torch.device | str | None, optional
        使用デバイス。None の場合は "cuda" が利用可能なら CUDA、そうでなければ CPU を選択する。

    Notes
    -----
    - seed は初期化時に設定される（再現性は DataLoader 側の worker seed 等にも依存する）。
    - Scheduler は fit() 内で初期化され、resume 時は checkpoint から状態を復元する。
    - 本 Trainer は DDP を想定しない（単一プロセス前提）。
    """

    def __init__(
        self,
        *,
        cfg: SimCLRTrainerConfig | None = None,
        aug_cfg: SimCLRAugConfig | None = None,
        model: SimCLRModel | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg or SimCLRTrainerConfig()
        self.aug_cfg = aug_cfg or SimCLRAugConfig()
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        _set_seed(self.cfg.seed)

        self.model = model or SimCLRModel()
        self.model.to(self.device)

        self.criterion = NTXentLoss(NTXentLossConfig(temperature=self.cfg.temperature))
        self.criterion.to(self.device)

        self.optimizer = self._build_optimizer()

        self.out_dir = Path(self.cfg.out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        self._scheduler: WarmupCosineSchedule | None = None
        self._resume_payload: dict[str, object] | None = None

    def fit(
        self,
        train_loader: DataLoader, # type: ignore
        *,
        max_epochs: int | None = None,
        resume: str | Path | Literal["auto", "none"] | None = None,
    ) -> FitHistory:
        """学習を実行し、epoch 単位の履歴を返す。

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            学習データローダ。各 batch は Tensor 画像、または (images, ...) 形式のタプル/リストを想定する。
            画像 Tensor の shape は (B, C, H, W)。
        max_epochs : int | None, optional
            学習 epoch 数の上書き。None の場合は cfg.epochs を用いる。
        resume : {"auto", "none"} | str | Path | None, optional
            再開指定。
            - None: cfg.resume_from を用いる
            - "none": 再開しない
            - "auto": checkpoints/last.pt があればそれを、無ければ epoch_*.pt の最新を探す
            - str/Path: 指定パスの checkpoint から再開する

        Returns
        -------
        FitHistory
            学習履歴（epoch 単位）。training_history.json にも同内容を保存する。

        Raises
        ------
        TypeError
            train_loader が DataLoader でない場合。
        ValueError
            epochs が 1 未満、または train_loader が空の場合など、設定・入力が不正な場合。
        FileNotFoundError
            resume で明示指定した checkpoint が存在しない場合。
        KeyError
            再開時に必要なキーが checkpoint / history json に存在しない場合。

        Notes
        -----
        - lr は fit() 冒頭で「実効バッチに合わせた線形スケール」を適用し、その値を base_lrs として scheduler を構築する。
        - Scheduler の step() は optimizer step の直前に呼ばれる（実装仕様）。
        - history["epoch"] は 1 始まりで保存する（ログ表示・可視化用途の都合）。
        """
        if not isinstance(train_loader, DataLoader):  # type: ignore
            raise TypeError("train_loader must be a torch.utils.data.DataLoader")

        epochs = int(max_epochs) if max_epochs is not None else int(self.cfg.epochs)
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1. got {epochs}")

        steps_per_epoch = self._steps_per_epoch(train_loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * int(self.cfg.warmup_epochs)

        # resume
        resume_spec = resume if resume is not None else self.cfg.resume_from
        start_epoch = 0
        if resume_spec != "none":
            if self._try_resume_from(resume_spec):
                start_epoch = int(self.epoch) + 1
                if start_epoch >= epochs:
                    return {"epoch": [], "loss": [], "lr": [], "pos_sim": [], "neg_sim": []}

        # base lr scaling
        target_lr = self._scaled_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = float(target_lr)

        self._scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lrs=[float(target_lr) for _ in self.optimizer.param_groups],
        )

        # restore scheduler
        if self._resume_payload is not None:
            st = self._resume_payload.get("scheduler")
            if isinstance(st, dict):
                self._scheduler.load_state_dict(st)
            else:
                self._scheduler.load_state_dict(
                    {
                        "warmup_steps": int(warmup_steps),
                        "total_steps": int(total_steps),
                        "base_lrs": [float(target_lr) for _ in self.optimizer.param_groups],
                        "step_idx": int(self.global_step),
                    }
                )

        history: FitHistory = {"epoch": [], "loss": [], "lr": [], "pos_sim": [], "neg_sim": []}

        # resume history from JSON (if exists)
        history_path = self.out_dir / "training_history.json"
        if start_epoch > 0 and history_path.exists():
            loaded = _load_json_dict(history_path)
            for k in ("epoch", "loss", "lr", "pos_sim", "neg_sim"):
                if k not in loaded:
                    raise KeyError(f"history json missing key: {k}")
                if not isinstance(loaded[k], list):
                    raise TypeError(f"history['{k}'] in json must be a list, got {type(loaded[k]).__name__}")
            history = loaded  # type: ignore[assignment]
        else:
            _atomic_write_json(history, history_path)  # type: ignore[arg-type]

        for ep in range(start_epoch, epochs):
            self.epoch = ep
            ep_metrics = self._train_one_epoch(train_loader)

            history["epoch"].append(int(ep+1))
            history["loss"].append(float(ep_metrics["loss"]))
            history["lr"].append(float(ep_metrics["lr"]))
            history["pos_sim"].append(float(ep_metrics["pos_sim"]))
            history["neg_sim"].append(float(ep_metrics["neg_sim"]))

            _atomic_write_json(history, history_path)  # type: ignore[arg-type]

            # checkpoints
            self._save_last_checkpoint()
            if self.cfg.save_every > 0 and ((ep + 1) % self.cfg.save_every == 0):
                self._save_checkpoint(tag=f"epoch_{ep+1:04d}")
                self._prune_checkpoints(keep_last_k=self.cfg.keep_last_k)

        return history

    def save(self, path: str | Path) -> None:
        """モデル重み（state_dict）を保存する。

        Parameters
        ----------
        path : str | pathlib.Path
            保存先パス。親ディレクトリは必要に応じて作成する。

        Notes
        -----
        - 保存するのは `self.model.state_dict()` のみであり、optimizer/scheduler 等は含まれない。
        学習再開用途には checkpoints/ 以下の checkpoint（last.pt / epoch_*.pt）を用いる。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        """モデル重み（state_dict）を読み込み、モデルに適用する。

        Parameters
        ----------
        path : str | pathlib.Path
            読み込む state_dict のパス（torch.save されたもの）。
        strict : bool, default=True
            state_dict の厳格一致を要求するかどうか（nn.Module.load_state_dict の strict）。

        Raises
        ------
        FileNotFoundError
            path が存在しない場合（torch.load 側で発生）。
        RuntimeError
            strict=True でキー不一致がある場合など（load_state_dict 由来）。

        Notes
        -----
        - optimizer/scheduler の状態は復元しない。学習再開が目的なら checkpoint を使用すること。
        - 読み込みは CPU に map してから適用する（GPU メモリ節約と互換性のため）。
        """
        sd = torch.load(Path(path), map_location="cpu")
        self.model.load_state_dict(sd, strict=strict)

    # ---------- internals ----------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lcfg = LARSConfig(
            lr=self.cfg.base_lr,
            momentum=self.cfg.momentum,
            use_nesterov=self.cfg.use_nesterov,
            weight_decay=self.cfg.weight_decay,
            exclude_from_weight_decay=None,
            exclude_from_layer_adaptation=None,
        )
        return LARS(self.model.named_parameters(), lcfg)

    def _steps_per_epoch(self, loader: DataLoader) -> int:  # type: ignore
        n_batches = len(loader)
        if n_batches < 1:
            raise ValueError("train_loader must have at least 1 batch")
        accum = int(self.cfg.accum_steps)
        if accum < 1:
            raise ValueError(f"accum_steps must be >=1. got {accum}")
        return (n_batches + accum - 1) // accum

    def _scaled_lr(self) -> float:
        eff_batch = int(self.cfg.batch_size) * int(self.cfg.accum_steps)
        if eff_batch < 1:
            raise ValueError(f"effective batch must be >=1. got {eff_batch}")
        return float(self.cfg.base_lr) * (float(eff_batch) / float(self.cfg.target_global_batch))

    @staticmethod
    def _pair_sims(z1: Tensor, z2: Tensor) -> tuple[float, float]:
        # z1,z2: (B,d) L2-normalized expected
        pos = float((z1 * z2).sum(dim=1).mean().detach().cpu())

        z_cat = torch.cat([z1, z2], dim=0)  # (2B,d)
        sim = z_cat @ z_cat.t()
        n = sim.shape[0]
        b = z1.shape[0]

        mask = torch.ones((n, n), device=sim.device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        idx = torch.arange(b, device=sim.device)
        mask[idx, idx + b] = False
        mask[idx + b, idx] = False
        neg = float(sim[mask].mean().detach().cpu())
        return pos, neg

    def _train_one_epoch(self, train_loader: DataLoader) -> dict[str, float]: # type: ignore
        self.model.train()
        if self._scheduler is None:
            raise RuntimeError("scheduler is not initialized. call fit()")

        t0 = time.time()

        accum = int(self.cfg.accum_steps)
        if accum < 1:
            raise ValueError(f"accum_steps must be >=1. got {accum}")

        self.optimizer.zero_grad(set_to_none=True)

        # epoch aggregates
        sum_loss = 0.0
        n_loss = 0

        sum_pos = 0.0
        sum_neg = 0.0
        n_sim = 0

        for step, batch in enumerate(train_loader):
            x = batch[0] if isinstance(batch, tuple | list) else batch
            if not isinstance(x, torch.Tensor):
                raise TypeError("train_loader must yield Tensor images or (images, ...) tuples")
            x = x.to(self.device, non_blocking=True)

            x1 = preprocess_for_train_batch(x, self.aug_cfg)
            x2 = preprocess_for_train_batch(x, self.aug_cfg)

            with _autocast_ctx(self.device, self.cfg.amp_dtype):
                z1 = self.model(x1)
                z2 = self.model(x2)
                z = torch.stack([z1, z2], dim=1).reshape(-1, z1.shape[-1])
                loss = self.criterion(z)

            # accumulation
            (loss / float(accum)).backward()

            sum_loss += float(loss.detach().cpu())
            n_loss += 1

            do_step = ((step + 1) % accum == 0) or (step + 1 == len(train_loader))
            if do_step:
                # step-wise scheduler then optimizer step
                self._scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # sims: use the last micro-batch in this accumulation window
                with torch.no_grad():
                    pos, neg = self._pair_sims(z1.detach(), z2.detach())
                sum_pos += pos
                sum_neg += neg
                n_sim += 1

        ep_sec = float(time.time() - t0)
        ep_loss = float(sum_loss / max(1, n_loss))
        ep_pos = float(sum_pos / max(1, n_sim))
        ep_neg = float(sum_neg / max(1, n_sim))
        ep_lr = float(self._scheduler.lr)

        # epoch summary log (print once per epoch)
        print(
            f"[epoch={self.epoch+1:04d}] "
            f"loss={ep_loss:.4f} lr={ep_lr:.6g} pos_sim={ep_pos:.5f} neg_sim={ep_neg:.5f} sec={ep_sec:.1f}"
        )

        return {
            "loss": ep_loss,
            "lr": ep_lr,
            "pos_sim": ep_pos,
            "neg_sim": ep_neg,
            "sec": ep_sec,
        }

    def _try_resume_from(self, resume_spec: str | Path | Literal["auto", "none"]) -> bool:
        if resume_spec == "none":
            self._resume_payload = None
            return False

        path: Path | None
        if resume_spec == "auto":
            path = self._find_auto_resume_checkpoint()
            if path is None:
                self._resume_payload = None
                return False
        else:
            path = Path(resume_spec)
            if not path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {path}")

        payload = _torch_load_checkpoint(path)

        if "model" not in payload:
            raise KeyError("checkpoint missing: model")
        self.model.load_state_dict(payload["model"])  # type: ignore[arg-type]

        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]

        self.epoch = int(payload.get("epoch", 0))  # type: ignore[arg-type]
        self.global_step = int(payload.get("global_step", 0))  # type: ignore[arg-type]

        cfg_obj = payload.get("cfg")
        if isinstance(cfg_obj, dict):
            self.cfg = SimCLRTrainerConfig(**cfg_obj)
        elif isinstance(cfg_obj, SimCLRTrainerConfig):
            self.cfg = cfg_obj

        aug_obj = payload.get("aug_cfg")
        if isinstance(aug_obj, dict):
            self.aug_cfg = SimCLRAugConfig(**aug_obj)
        elif isinstance(aug_obj, SimCLRAugConfig):
            self.aug_cfg = aug_obj

        self._resume_payload = payload
        return True

    def _find_auto_resume_checkpoint(self) -> Path | None:
        last = self.ckpt_dir / "last.pt"
        if last.exists():
            return last
        pts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return pts[-1] if len(pts) else None

    def _atomic_save(self, payload: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def _save_last_checkpoint(self) -> None:
        payload: dict[str, object] = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": asdict(self.cfg),
            "aug_cfg": asdict(self.aug_cfg),
        }
        if self._scheduler is not None:
            payload["scheduler"] = self._scheduler.state_dict()
        self._atomic_save(payload, self.ckpt_dir / "last.pt")

    def _save_checkpoint(self, *, tag: str) -> None:
        path = self.ckpt_dir / f"{tag}.pt"
        payload: dict[str, object] = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": asdict(self.cfg),
            "aug_cfg": asdict(self.aug_cfg),
        }
        if self._scheduler is not None:
            payload["scheduler"] = self._scheduler.state_dict()

        self._atomic_save(payload, path)
        self._atomic_save(payload, self.ckpt_dir / "last.pt")

    def _prune_checkpoints(self, *, keep_last_k: int) -> None:
        if keep_last_k <= 0:
            return
        pts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if len(pts) <= keep_last_k:
            return
        for p in pts[:-keep_last_k]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass