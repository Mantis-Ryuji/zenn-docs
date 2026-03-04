from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_fit_history_json(*, out_dir: str | Path, filename: str = "training_history.json") -> dict[str, Any]:
    p = Path(out_dir) / filename
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"history json must be a dict, got {type(data).__name__}")
    return data


def plot_training_history(history: Mapping[str, Any], *, title: str | None = None) -> None:
    def as_int_list(x: Any, name: str) -> list[int]:
        if not isinstance(x, Sequence) or isinstance(x, (str | bytes)):
            raise TypeError(f"history['{name}'] must be a sequence, got {type(x).__name__}.")
        return [int(v) for v in x]

    def as_float_list(x: Any, name: str) -> list[float]:
        if not isinstance(x, Sequence) or isinstance(x, (str | bytes)):
            raise TypeError(f"history['{name}'] must be a sequence, got {type(x).__name__}.")
        return [float(v) for v in x]

    epoch = as_int_list(history["epoch"], "epoch")
    loss = as_float_list(history["loss"], "loss")
    lr = as_float_list(history["lr"], "lr")
    pos_sim = as_float_list(history["pos_sim"], "pos_sim")
    neg_sim = as_float_list(history["neg_sim"], "neg_sim")

    n = len(epoch)
    if not (len(loss) == len(lr) == len(pos_sim) == len(neg_sim) == n):
        raise ValueError(
            "Length mismatch: "
            f"epoch={len(epoch)}, loss={len(loss)}, lr={len(lr)}, pos_sim={len(pos_sim)}, neg_sim={len(neg_sim)}"
        )

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True)

    ax00 = axs[0, 0]
    ax00.plot(epoch, loss, label="loss")
    ax00.set_xlabel("epoch")
    ax00.set_ylabel("loss")
    ax00.set_title(f"{title} : loss" if title else "loss")
    ax00.legend()

    ax01 = axs[0, 1]
    ax01.plot(epoch, lr, label="lr")
    ax01.set_xlabel("epoch")
    ax01.set_ylabel("lr")
    ax01.set_title(f"{title} : lr" if title else "lr")
    ax01.legend()

    ax10 = axs[1, 0]
    ax10.plot(epoch, pos_sim, label="pos_sim")
    ax10.set_xlabel("epoch")
    ax10.set_ylabel("pos_sim")
    ax10.set_title(f"{title} : pos_sim" if title else "pos_sim")
    ax10.legend()

    ax11 = axs[1, 1]
    ax11.plot(epoch, neg_sim, label="neg_sim")
    ax11.set_xlabel("epoch")
    ax11.set_ylabel("neg_sim")
    ax11.set_title(f"{title} : neg_sim" if title else "neg_sim")
    ax11.legend()

    fig.tight_layout()
    plt.show()