from __future__ import annotations

import math
import time
from dataclasses import dataclass
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

# tqdm は optional（無ければ通常 print ログにフォールバック）
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

AmpDType = Literal["bf16", "fp16", "none"]


class FitHistory(TypedDict):
    """fit() が返す学習ログ。

    Notes
    -----
    - 1 epoch ごとに 1 要素を append する。
    - `train_loss` は 1 epoch 内の **ミニバッチ損失（accum 前の loss）** の平均。
    - `lr` は epoch 終了時点の scheduler の LR（先頭 param group）。
    """

    epoch: list[int]
    global_step: list[int]
    train_loss: list[float]
    lr: list[float]
    sec: list[float]


@dataclass(frozen=True)
class SimCLRTrainerConfig:
    """SimCLR 学習の Trainer 設定。

    Notes
    -----
    - SimCLR の典型設定（ImageNet / batch=4096 / lr=4.8）を基準にする。
      `base_lr` は「global batch=4096 相当」の学習率として扱い、
      実際の global batch（= batch_size * world_size * accum_steps）に対して線形スケールする。
    - scheduler は「最初の warmup_epochs は線形 warmup」→「残りは Cosine Annealing（restart なし）」。
      いずれも **step 単位**で更新する。
    """

    # train loop
    epochs: int = 1000
    warmup_epochs: int = 10
    log_every: int = 5  # steps
    eval_every: int = 5  # epochs (0 disables)
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
    temperature: float = 0.5

    # AMP
    amp_dtype: AmpDType = "bf16"

    # misc
    out_dir: Path = Path("runs/simclr")
    save_every: int = 5  # epochs
    keep_last_k: int = 1  # checkpoints

    # resume
    resume_from: str | Path | Literal["auto", "none"] = "auto"  # checkpoints


class SimCLRModel(nn.Module):
    """Encoder + ProjectionHead の SimCLR モデル。"""

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
        h = self.encoder(x)          # (B, 2048)
        z = self.projector(h)        # (B, 128) (l2-normalized by default)
        return z


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_ctx(device: torch.device, amp_dtype: AmpDType):
    if amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type != "cuda":
        # CPU/MPS で bf16 autocast を使いたい場合もあるが、挙動差が出やすいので明示的に無効化。
        return torch.autocast(device_type=device.type, enabled=False)

    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown amp_dtype: {amp_dtype}")


class WarmupCosineSchedule:
    """線形 warmup → cosine annealing（restart なし）の step-wise scheduler。

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        対象 optimizer。
    warmup_steps : int
        warmup の総 step 数。
    total_steps : int
        学習全体の総 step 数。
    base_lrs : list[float]
        step=0 以降に到達したい「目標 LR」（param group ごと）。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int, base_lrs: list[float]) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >=0. got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be >0. got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps must be < total_steps. got warmup_steps={warmup_steps}, total_steps={total_steps}")
        if len(base_lrs) != len(optimizer.param_groups):
            raise ValueError("base_lrs length must match optimizer.param_groups")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = base_lrs
        self.step_idx = 0

    def step(self) -> None:
        s = self.step_idx
        if s < self.warmup_steps:
            # linear warmup: 0 -> base_lr
            scale = float(s + 1) / float(self.warmup_steps)
        else:
            # cosine: base_lr -> 0
            t = float(s - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            # t in [0,1)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            g["lr"] = base_lr * scale

        self.step_idx += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps),
            "base_lrs": [float(x) for x in self.base_lrs],
            "step_idx": int(self.step_idx),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        # fail fast
        warmup_steps = int(state["warmup_steps"])  # type: ignore[arg-type]
        total_steps = int(state["total_steps"])    # type: ignore[arg-type]
        base_lrs = list(state["base_lrs"])         # type: ignore[arg-type]
        step_idx = int(state["step_idx"])          # type: ignore[arg-type]

        if warmup_steps < 0 or total_steps <= 0 or warmup_steps >= total_steps:
            raise ValueError("Invalid scheduler state.")
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler state base_lrs length mismatch.")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [float(x) for x in base_lrs]
        self.step_idx = step_idx

        # restore lr consistent with current step
        # (do not increment step_idx)
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
        return float(self.optimizer.param_groups[0]["lr"])


class SimCLRTrainer:
    """SimCLRTrainer（単一プロセス + 勾配蓄積対応）。

    Notes
    -----
    - DDP/FSDP はこのファイルでは扱わない（必要なら外側で wrap する）。
    - DataLoader は (images, labels) あるいは images のみを返してよい（labels は無視する）。
    - augmentation は `preprocess_for_train_batch` を 2 回呼ぶことで 2-view を生成する。
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
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ---------- public API ----------

    def fit(self, train_loader: DataLoader, *, max_epochs: int | None = None, resume: str | Path | Literal["auto", "none"] | None = None) -> FitHistory:  # type: ignore
        if not isinstance(train_loader, DataLoader): # type: ignore
            raise TypeError("train_loader must be a torch.utils.data.DataLoader")

        epochs = int(max_epochs) if max_epochs is not None else int(self.cfg.epochs)
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1. got {epochs}")

        steps_per_epoch = self._steps_per_epoch(train_loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * int(self.cfg.warmup_epochs)


        # resume (auto/explicit)
        resume_spec = resume if resume is not None else self.cfg.resume_from
        start_epoch = 0
        if resume_spec != "none":
            resumed = self._try_resume_from(resume_spec)
            if resumed:
                # continue from next epoch (checkpoint epoch is "completed epoch index")
                start_epoch = int(self.epoch) + 1
                # if already finished, return empty history with model loaded
                if start_epoch >= epochs:
                    return {
                        "epoch": [],
                        "global_step": [],
                        "train_loss": [],
                        "lr": [],
                        "sec": [],
                    }

        # (re-)compute scheduler based on the *requested* total epochs

        # base lr scaling by global batch (accum included)
        target_lr = self._scaled_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = target_lr

        self._scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lrs=[target_lr for _ in self.optimizer.param_groups],
        )

        # restore scheduler position if resumed
        if self._resume_payload is not None and self._scheduler is not None: # type: ignore
            st = self._resume_payload.get("scheduler")
            if isinstance(st, dict):
                self._scheduler.load_state_dict(st)  # type: ignore[arg-type]
            else:
                # fallback: align scheduler to global_step
                self._scheduler.load_state_dict(
                    {
                        "warmup_steps": int(warmup_steps),
                        "total_steps": int(total_steps),
                        "base_lrs": [float(target_lr) for _ in self.optimizer.param_groups],
                        "step_idx": int(self.global_step),
                    }
                )

        history: FitHistory = {
            "epoch": [],
            "global_step": [],
            "train_loss": [],
            "lr": [],
            "sec": [],
        }

        for ep in range(start_epoch, epochs):
            self.epoch = ep
            ep_loss, ep_sec = self._train_one_epoch(train_loader)
            history["epoch"].append(int(ep))
            history["global_step"].append(int(self.global_step))
            history["train_loss"].append(float(ep_loss))
            history["lr"].append(float(self._scheduler.lr))
            history["sec"].append(float(ep_sec))

            # always update last.pt for auto-resume
            self._save_last_checkpoint()


            if self.cfg.save_every > 0 and ((ep + 1) % self.cfg.save_every == 0):
                self._save_checkpoint(tag=f"epoch_{ep+1:04d}")
                self._prune_checkpoints(keep_last_k=self.cfg.keep_last_k)


        return history

    def save(self, path: str | Path) -> None:
        """重みのみ保存。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        """重みのみ読み込み。"""
        sd = torch.load(Path(path), map_location="cpu")
        self.model.load_state_dict(sd, strict=strict)

    # ---------- internals ----------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # LARS はパラメータ名で除外設定を持つため named_parameters を渡す設計
        lcfg = LARSConfig(
            lr=self.cfg.base_lr,
            momentum=self.cfg.momentum,
            use_nesterov=self.cfg.use_nesterov,
            weight_decay=self.cfg.weight_decay,
            exclude_from_weight_decay=None,  # default: bias & LayerNorm系
            exclude_from_layer_adaptation=None,
        )
        return LARS(self.model.named_parameters(), lcfg)

    def _steps_per_epoch(self, loader: DataLoader) -> int: # type: ignore
        # gradient accumulation を考慮した optimizer step 数（= ceil(num_batches / accum_steps)）
        n_batches = len(loader)
        if n_batches < 1:
            raise ValueError("train_loader must have at least 1 batch")
        accum = int(self.cfg.accum_steps)
        if accum < 1:
            raise ValueError(f"accum_steps must be >=1. got {accum}")
        return (n_batches + accum - 1) // accum

    def _scaled_lr(self) -> float:
        # SimCLR: lr scales linearly with batch size
        # base_lr is for target_global_batch.
        eff_batch = int(self.cfg.batch_size) * int(self.cfg.accum_steps)
        if eff_batch < 1:
            raise ValueError(f"effective batch must be >=1. got {eff_batch}")
        return float(self.cfg.base_lr) * (float(eff_batch) / float(self.cfg.target_global_batch))
    def _train_one_epoch(self, train_loader: DataLoader) -> tuple[float, float]:  # type: ignore
        self.model.train()

        t_epoch0 = time.time()
        # epoch-level aggregates (loss is *unscaled* NT-Xent per minibatch)
        sum_loss = 0.0
        n_loss = 0

        # log_every window (for tqdm/print)
        win_t0 = time.time()
        win_loss = 0.0
        win_n = 0

        self.optimizer.zero_grad(set_to_none=True)

        accum = int(self.cfg.accum_steps)
        if accum < 1:
            raise ValueError(f"accum_steps must be >=1. got {accum}")

        use_tqdm = (tqdm is not None)
        pbar = tqdm(train_loader, desc=f"train epoch {self.epoch+1:04d}", leave=False) if use_tqdm else train_loader # type: ignore

        for step, batch in enumerate(pbar):
            x = batch[0] if isinstance(batch, tuple | list) else batch
            if not isinstance(x, torch.Tensor):
                raise TypeError("train_loader must yield Tensor images or (images, ...) tuples")
            x = x.to(self.device, non_blocking=True)

            # 2 views with independent stochasticity
            x1 = preprocess_for_train_batch(x, self.aug_cfg)
            x2 = preprocess_for_train_batch(x, self.aug_cfg)

            with _autocast_ctx(self.device, self.cfg.amp_dtype):
                z1 = self.model(x1)
                z2 = self.model(x2)

                # interleave to satisfy NTXentLoss positive-pair layout: (0,1),(2,3),...
                z = torch.stack([z1, z2], dim=1).reshape(-1, z1.shape[-1])
                loss = self.criterion(z)

            # gradient accumulation: scale loss so that effective gradient matches large batch
            loss_scaled = loss / float(accum)
            loss_scaled.backward()

            loss_val = float(loss.detach().cpu())
            sum_loss += loss_val
            n_loss += 1

            win_loss += loss_val
            win_n += 1

            do_step = ((step + 1) % accum == 0) or (step + 1 == len(train_loader))
            if do_step:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self._scheduler is None:
                    raise RuntimeError("scheduler is not initialized. call fit()")
                self._scheduler.step()

                self.global_step += 1

                if self.cfg.log_every > 0 and (self.global_step % self.cfg.log_every == 0):
                    dt = time.time() - win_t0
                    mean_loss = win_loss / max(1, win_n)
                    lr = float(self._scheduler.lr)

                    if use_tqdm:
                        # tqdm の postfix に反映（print はしない）
                        try:
                            pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.3g}", sec=f"{dt:.1f}") # type: ignore
                        except Exception:
                            pass
                    else:
                        print(
                            f"[epoch {self.epoch+1:04d}] step={self.global_step:07d} "
                            f"loss={mean_loss:.4f} lr={lr:.6g} ({dt:.1f}s)"
                        )

                    win_loss = 0.0
                    win_n = 0
                    win_t0 = time.time()

        ep_sec = time.time() - t_epoch0
        ep_loss = sum_loss / max(1, n_loss)

        if not use_tqdm:
            print(f"[epoch {self.epoch+1:04d}] end: loss={ep_loss:.4f} sec={ep_sec:.1f}")

        return float(ep_loss), float(ep_sec)


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

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload must be a dict")

        # restore (fail fast where possible)
        if "model" not in payload:
            raise KeyError("checkpoint missing: model")
        self.model.load_state_dict(payload["model"])  # type: ignore[arg-type]

        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]

        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))

        # optional: cfg/aug_cfg restore (useful for true resume)
        if "cfg" in payload and isinstance(payload["cfg"], SimCLRTrainerConfig):
            self.cfg = payload["cfg"]  # type: ignore[assignment]
        if "aug_cfg" in payload and isinstance(payload["aug_cfg"], SimCLRAugConfig):
            self.aug_cfg = payload["aug_cfg"]  # type: ignore[assignment]

        self._resume_payload = payload  # type: ignore[assignment]
        return True

    def _find_auto_resume_checkpoint(self) -> Path | None:
        last = self.ckpt_dir / "last.pt"
        if last.exists():
            return last

        # fallback: latest epoch_*.pt
        pts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if len(pts) == 0:
            return None
        return pts[-1]

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
            "cfg": self.cfg,
            "aug_cfg": self.aug_cfg,
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
                "cfg": self.cfg,
                "aug_cfg": self.aug_cfg,
            }
            if self._scheduler is not None:
                payload["scheduler"] = self._scheduler.state_dict()

            self._atomic_save(payload, path)

            # always update last.pt for auto-resume
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