from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch import Tensor

CROP_PROPORTION: float = 0.875
BrightnessImpl = Literal["simclrv1", "simclrv2"]


@dataclass(frozen=True)
class SimCLRAugConfig:
    """SimCLR 画像前処理（バッチ版）の設定。

    本モジュールは torchvision.transforms を 1 サンプルずつ回すのではなく、(B,C,H,W) の
    **バッチ Tensor に対してベクトル化した拡張**を適用する前提で設計されている。
    学習用前処理（crop/flip/color distortion/blur）と、評価用前処理（resize + center crop）で
    使い回すパラメータをまとめる。

    Parameters
    ----------
    out_h : int, default=256
        出力画像の高さ。
    out_w : int, default=256
        出力画像の幅。
    color_jitter_strength : float, default=1.0
        Color jitter の強度スケール。
        brightness / contrast / saturation は 0.8 * strength、hue は 0.2 * strength を上限として使う。
    impl : {"simclrv2", "simclrv1"}, default="simclrv2"
        brightness の実装流儀。
        - "simclrv2": 乗算（factor を掛ける）方式
        - "simclrv1": 加算（delta を足す）方式
    crop : bool, default=True
        学習時に random resized crop を行うかどうか。
    flip : bool, default=True
        学習時に左右反転を行うかどうか。
    color_distort : bool, default=True
        学習時に color jitter + grayscale を行うかどうか。
    blur : bool, default=True
        学習時に Gaussian blur を行うかどうか。
    blur_prob : float, default=0.5
        blur の適用確率（各サンプル独立）。

    Notes
    -----
    - すべての前処理は (B,C,H,W) の Tensor 入力を想定し、値域は [0,1] にクリップして扱う実装を想定する。
    - blur_prob は「各サンプル独立」の Bernoulli で判定される（バッチ一括ではない）。
    - out_h/out_w は学習・評価の両方で最終解像度として利用される。
    """

    out_h: int = 256
    out_w: int = 256
    color_jitter_strength: float = 1.0
    impl: BrightnessImpl = "simclrv2"
    crop: bool = True
    flip: bool = True
    color_distort: bool = True
    blur: bool = True
    blur_prob: float = 0.5


def _as_float01_bchw(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor): # type: ignore
        raise TypeError(f"x must be torch.Tensor. got {type(x).__name__}")
    if x.ndim != 4:
        raise ValueError(f"x must be 4D (B,C,H,W). got shape={tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"expected C=3. got C={x.shape[1]}")

    if x.dtype == torch.uint8:
        x = x.float().div(255.0)
    else:
        x = x.float()

    return x.clamp(0.0, 1.0)


def _rand_bool_mask(bsz: int, p: float, *, device: torch.device) -> Tensor:
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be in [0,1]. got {p}")
    return (torch.rand((bsz,), device=device) < float(p)).to(torch.bool)


def _apply_where(mask_b: Tensor, x_new: Tensor, x_old: Tensor) -> Tensor:
    if mask_b.ndim != 1:
        raise ValueError(f"mask must be 1D (B,). got shape={tuple(mask_b.shape)}")
    m = mask_b[:, None, None, None]
    return torch.where(m, x_new, x_old)


# -------------------------
# Color jitter primitives (batch-safe)
# -------------------------
def _random_brightness(x: Tensor, max_delta: float, *, impl: BrightnessImpl) -> Tensor:
    bsz = x.shape[0]
    device = x.device
    if float(max_delta) <= 0.0:
        return x

    if impl == "simclrv2":
        lo = max(1.0 - float(max_delta), 0.0)
        hi = 1.0 + float(max_delta)
        factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(lo, hi)
        return (x * factor).clamp(0.0, 1.0)

    # impl == "simclrv1"
    delta = torch.empty((bsz, 1, 1, 1), device=device).uniform_(-float(max_delta), float(max_delta))
    return (x + delta).clamp(0.0, 1.0)


def _random_contrast(x: Tensor, max_delta: float) -> Tensor:
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(1.0 - float(max_delta), 1.0 + float(max_delta))
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * factor + mean).clamp(0.0, 1.0)


def _random_saturation(x: Tensor, max_delta: float) -> Tensor:
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(1.0 - float(max_delta), 1.0 + float(max_delta))
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
    gray3 = gray.expand(-1, 3, -1, -1)
    return (x * factor + gray3 * (1.0 - factor)).clamp(0.0, 1.0)


def _rgb_to_hsv(x: Tensor) -> Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    v = maxc
    delt = maxc - minc

    s = torch.where(maxc > 0, delt / (maxc + 1e-12), torch.zeros_like(maxc))

    denom = delt + 1e-12
    rc = (maxc - r) / denom
    gc = (maxc - g) / denom
    bc = (maxc - b) / denom

    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r) & (delt > 0), (bc - gc), h)
    h = torch.where((maxc == g) & (delt > 0), (2.0 + rc - bc), h)
    h = torch.where((maxc == b) & (delt > 0), (4.0 + gc - rc), h)
    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: Tensor) -> Tensor:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h6 = (h * 6.0) % 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i.float()

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.where(
        i == 0,
        v,
        torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))),
    )
    g = torch.where(
        i == 0,
        t,
        torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))),
    )
    b = torch.where(
        i == 0,
        p,
        torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))),
    )
    return torch.stack([r, g, b], dim=1)


def _random_hue(x: Tensor, max_delta: float) -> Tensor:
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    delta = torch.empty((bsz, 1, 1), device=device).uniform_(-float(max_delta), float(max_delta))
    hsv = _rgb_to_hsv(x)
    h = (hsv[:, 0] + delta).remainder(1.0)
    return _hsv_to_rgb(torch.stack([h, hsv[:, 1], hsv[:, 2]], dim=1)).clamp(0.0, 1.0)


def _to_grayscale_keep3(x: Tensor) -> Tensor:
    return TVF.rgb_to_grayscale(x, num_output_channels=3)


def color_jitter_batch(x: Tensor, *, strength: float, impl: BrightnessImpl) -> Tensor:
    """SimCLR の color jitter（バッチ版、サンプルごとに順序シャッフル）。

    brightness / contrast / saturation / hue の 4 変換を、それぞれ確率ではなく「必ず」適用するが、
    **各サンプルごとに適用順序をランダムにシャッフル**する。
    これにより、バッチ内で同一順序に固定されることを避け、SimCLR の実装流儀に近づける。

    Parameters
    ----------
    x : torch.Tensor
        入力画像。shape は (B, 3, H, W)、値域は [0,1] を想定する。
    strength : float
        jitter 強度。0 以下の場合は入力をそのまま返す。
    impl : {"simclrv1", "simclrv2"}
        brightness の実装流儀（加算/乗算）。

    Returns
    -------
    torch.Tensor
        jitter 後の画像。shape は (B, 3, H, W)。値域は [0,1] にクリップされる。

    Notes
    -----
    - サンプルごとの順序シャッフルは perm (B,4) を生成して実現する。
    - 各変換はバッチ演算として実装されており、Python ループは「4 変換の順序適用」に限定される。
    """
    s = float(strength)
    if s <= 0.0:
        return x

    b = 0.8 * s
    c = 0.8 * s
    sat = 0.8 * s
    h = 0.2 * s

    def f_b(xx: Tensor) -> Tensor:
        return _random_brightness(xx, b, impl=impl)

    def f_c(xx: Tensor) -> Tensor:
        return _random_contrast(xx, c)

    def f_s(xx: Tensor) -> Tensor:
        return _random_saturation(xx, sat)

    def f_h(xx: Tensor) -> Tensor:
        return _random_hue(xx, h)

    bsz = x.shape[0]
    device = x.device
    perm = torch.stack([torch.randperm(4, device=device) for _ in range(bsz)], dim=0)

    y = x
    for t in range(4):
        y0 = f_b(y)
        y1 = f_c(y)
        y2 = f_s(y)
        y3 = f_h(y)

        idx = perm[:, t]
        y = _apply_where(idx == 0, y0, y)
        y = _apply_where(idx == 1, y1, y)
        y = _apply_where(idx == 2, y2, y)
        y = _apply_where(idx == 3, y3, y)

    return y.clamp(0.0, 1.0)


def random_color_distort_batch(x: Tensor, *, strength: float, impl: BrightnessImpl, p: float = 1.0) -> Tensor:
    """SimCLR の color distortion（jitter + grayscale）を確率適用（バッチ版）。

    サンプルごとに確率 p で「color distortion」を適用する。
    color distortion は以下の合成から成る。
    - 確率 0.8 で color jitter
    - 確率 0.2 で grayscale（3ch のまま）

    Parameters
    ----------
    x : torch.Tensor
        入力画像。shape は (B, 3, H, W)、値域は [0,1] を想定する。
    strength : float
        color jitter の強度。
    impl : {"simclrv1", "simclrv2"}
        brightness の実装流儀。
    p : float, default=1.0
        distortion 全体を適用する確率（各サンプル独立）。

    Returns
    -------
    torch.Tensor
        変換後の画像。shape は (B, 3, H, W)。値域は [0,1] にクリップされる。

    Raises
    ------
    ValueError
        p が [0,1] でない場合（内部の乱数マスク生成で検証される）。

    Notes
    -----
    - p によるマスク（m_all）で対象サンプルを決め、jitter/grayscale はその subset に対してさらに独立判定する。
    - いずれの変換も「対象サンプルのみを差し替える」方式（torch.where）でバッチを維持する。
    """
    bsz = x.shape[0]
    device = x.device

    m_all = _rand_bool_mask(bsz, p, device=device)
    if not m_all.any().item():
        return x

    y = x

    m_j = _rand_bool_mask(bsz, 0.8, device=device) & m_all
    if m_j.any().item():
        y_j = color_jitter_batch(y, strength=strength, impl=impl)
        y = _apply_where(m_j, y_j, y)

    m_g = _rand_bool_mask(bsz, 0.2, device=device) & m_all
    if m_g.any().item():
        y_g = _to_grayscale_keep3(y)
        y = _apply_where(m_g, y_g, y)

    return y.clamp(0.0, 1.0)


# -------------------------
# Crop / flip / blur
# -------------------------
def _center_crop_and_resize_batch(x: Tensor, out_h: int, out_w: int, crop_proportion: float) -> Tensor:
    _, _, h, w = x.shape
    aspect = out_w / out_h

    iw = float(w)
    ih = float(h)
    if aspect > iw / ih:
        crop_h = int(round(crop_proportion / aspect * iw))
        crop_w = int(round(crop_proportion * iw))
    else:
        crop_h = int(round(crop_proportion * ih))
        crop_w = int(round(crop_proportion * aspect * ih))

    crop_h = max(1, min(crop_h, h))
    crop_w = max(1, min(crop_w, w))

    off_y = ((h - crop_h) + 1) // 2
    off_x = ((w - crop_w) + 1) // 2

    y = x[:, :, off_y : off_y + crop_h, off_x : off_x + crop_w]
    y = F.interpolate(y, size=(out_h, out_w), mode="bicubic", align_corners=False)
    return y.clamp(0.0, 1.0)


def random_resized_crop_batch(
    x: Tensor,
    *,
    out_h: int,
    out_w: int,
    area_range: tuple[float, float] = (0.08, 1.0),
    aspect_ratio_range: tuple[float, float] = (0.75, 1.33),
    p: float = 1.0,
) -> Tensor:
    """RandomResizedCrop 相当の処理をバッチで行う。

    サンプルごとにランダムな crop 矩形（面積比・アスペクト比）を生成し、指定解像度にリサイズする。
    実装は grid_sample を用いたベクトル化で、サンプルごとの crop を 1 回の演算で処理する。

    Parameters
    ----------
    x : torch.Tensor
        入力画像。shape は (B, C, H, W)、値域は [0,1] を想定する。
    out_h : int
        出力高さ。
    out_w : int
        出力幅。
    area_range : tuple[float, float], default=(0.08, 1.0)
        crop 面積比の範囲（各サンプル独立に一様サンプリング）。
    aspect_ratio_range : tuple[float, float], default=(0.75, 1.33)
        crop アスペクト比の範囲（log-space で一様サンプリング）。
    p : float, default=1.0
        適用確率（各サンプル独立）。適用されないサンプルは入力をそのまま返す。

    Returns
    -------
    torch.Tensor
        出力画像。shape は (B, C, out_h, out_w)。値域は [0,1] にクリップされる。

    Raises
    ------
    ValueError
        p が [0,1] でない場合（内部の乱数マスク生成で検証される）。

    Notes
    -----
    - apply マスクで対象サンプルのみ差し替えるため、バッチ全体の shape は維持される。
    - grid_sample は align_corners=True を使用している（座標変換の仕様に依存するため、必要なら統一すること）。
    - padding_mode="zeros" のため、crop が画像外に出る場合（理論上は clamp 済みで稀）には 0 埋めとなる。
    """
    b, _, h, w = x.shape
    device = x.device

    m = _rand_bool_mask(b, p, device=device)
    if not m.any().item():
        return x

    area = torch.empty((b,), device=device).uniform_(area_range[0], area_range[1])
    log_ar = torch.empty((b,), device=device).uniform_(
        math.log(aspect_ratio_range[0]), math.log(aspect_ratio_range[1])
    )
    ar = torch.exp(log_ar)

    target_area = area * float(h * w)
    crop_w = torch.round(torch.sqrt(target_area * ar)).to(torch.long)
    crop_h = torch.round(torch.sqrt(target_area / ar)).to(torch.long)

    crop_w = crop_w.clamp(1, w)
    crop_h = crop_h.clamp(1, h)

    max_off_x = (w - crop_w).clamp(min=0)
    max_off_y = (h - crop_h).clamp(min=0)

    off_x = (torch.rand((b,), device=device) * (max_off_x + 1).float()).floor().to(torch.long)
    off_y = (torch.rand((b,), device=device) * (max_off_y + 1).float()).floor().to(torch.long)

    yy = torch.linspace(0, 1, steps=out_h, device=device)[None, :, None]
    xx = torch.linspace(0, 1, steps=out_w, device=device)[None, None, :]

    crop_h_f = (crop_h - 1).clamp(min=1).float()[:, None, None]
    crop_w_f = (crop_w - 1).clamp(min=1).float()[:, None, None]
    y_pix = off_y.float()[:, None, None] + yy * crop_h_f
    x_pix = off_x.float()[:, None, None] + xx * crop_w_f

    y_pix = y_pix.expand(b, out_h, out_w)
    x_pix = x_pix.expand(b, out_h, out_w)

    y_norm = (y_pix / max(h - 1, 1)) * 2.0 - 1.0
    x_norm = (x_pix / max(w - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)

    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True).clamp(0.0, 1.0)
    return _apply_where(m, y, x)


def random_hflip_batch(x: Tensor, p: float = 0.5) -> Tensor:
    """左右反転を確率適用（バッチ版）。

    Parameters
    ----------
    x : torch.Tensor
        入力画像。shape は (B, C, H, W)。
    p : float, default=0.5
        左右反転を適用する確率（各サンプル独立）。

    Returns
    -------
    torch.Tensor
        変換後の画像。shape は (B, C, H, W)。

    Raises
    ------
    ValueError
        p が [0,1] でない場合（内部の乱数マスク生成で検証される）。

    Notes
    -----
    - 反転は `torch.flip(..., dims=[3])` を用い、対象サンプルのみ torch.where で差し替える。
    """
    b = x.shape[0]
    m = _rand_bool_mask(b, p, device=x.device)
    if not m.any().item():
        return x
    y = torch.flip(x, dims=[3])
    return _apply_where(m, y, x)


def gaussian_blur_batch(
    x: Tensor,
    *,
    kernel_size: int,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    p: float = 1.0,
) -> Tensor:
    """Gaussian blur を確率適用（バッチ版、サンプルごとに sigma をランダム化）。

    サンプルごとに sigma を一様サンプリングし、depthwise conv（groups=b*c）で
    分離可能な 1 次元カーネル（水平→垂直）として畳み込みを行う。

    Parameters
    ----------
    x : torch.Tensor
        入力画像。shape は (B, C, H, W)、値域は [0,1] を想定する。
    kernel_size : int
        カーネルサイズ。偶数が与えられた場合は +1 して奇数に丸める。
    sigma_min : float, default=0.1
        sigma の下限。
    sigma_max : float, default=2.0
        sigma の上限。
    p : float, default=1.0
        blur を適用する確率（各サンプル独立）。

    Returns
    -------
    torch.Tensor
        変換後の画像。shape は (B, C, H, W)。値域は [0,1] にクリップされる。

    Raises
    ------
    ValueError
        kernel_size < 1 の場合。
    ValueError
        p が [0,1] でない場合（内部の乱数マスク生成で検証される）。

    Notes
    -----
    - サンプルごとに異なるカーネルを持つため、入力を (1, B*C, H, W) に reshape し、
      groups=B*C の depthwise conv として処理する。
    - 出力は対象サンプルのみ差し替える（torch.where）。
    """
    b, c, h, w = x.shape
    device = x.device

    m = _rand_bool_mask(b, p, device=device)
    if not m.any().item():
        return x

    k = int(kernel_size)
    if k < 1:
        raise ValueError(f"kernel_size must be >=1. got {kernel_size}")
    if (k % 2) == 0:
        k += 1
    radius = k // 2

    sigma = torch.empty((b,), device=device).uniform_(float(sigma_min), float(sigma_max))

    xs = torch.arange(-radius, radius + 1, device=device).float()[None, :]
    sig = sigma[:, None].clamp(min=1e-6)
    ker = torch.exp(-(xs**2) / (2.0 * (sig**2)))
    ker = ker / ker.sum(dim=1, keepdim=True)

    inp = x.reshape(1, b * c, h, w)

    ker_h = ker[:, None, None, :].repeat_interleave(c, dim=0)
    ker_v = ker[:, None, :, None].repeat_interleave(c, dim=0)

    pad = radius
    out = F.conv2d(inp, ker_h, bias=None, stride=1, padding=(0, pad), groups=b * c)
    out = F.conv2d(out, ker_v, bias=None, stride=1, padding=(pad, 0), groups=b * c)
    out = out.reshape(b, c, h, w).clamp(0.0, 1.0)

    return _apply_where(m, out, x)


# -------------------------
# Public preprocess API (batch)
# -------------------------
def preprocess_for_train_batch(x: Tensor, cfg: SimCLRAugConfig) -> Tensor:
    """学習時の前処理（バッチ版）。

    入力を float / [0,1] / (B,3,H,W) に正規化した上で、設定に応じて以下を適用する。
    - random resized crop（常に適用）
    - random horizontal flip（p=0.5）
    - color distortion（常に適用、内部で jitter/grayscale は確率）
    - Gaussian blur（cfg.blur_prob で確率適用）

    Parameters
    ----------
    x : torch.Tensor
        入力画像。uint8 (0..255) または float を許容する。
        shape は (B, 3, H, W)。
    cfg : SimCLRAugConfig
        前処理設定。

    Returns
    -------
    torch.Tensor
        前処理後の画像。shape は (B, 3, cfg.out_h, cfg.out_w)。
        値域は [0,1] にクリップされる。

    Raises
    ------
    TypeError
        x が torch.Tensor でない場合。
    ValueError
        - x が 4 次元でない場合、または C!=3 の場合（内部の正規化関数で検証される）
        - cfg.impl が未サポートの場合

    Notes
    -----
    - 乱数は内部で torch.rand / torch.randperm を用いるため、再現性は RNG 状態に依存する。
      DataLoader worker や torch.manual_seed の設定と合わせて管理すること。
    - 前処理は dtype/device の変換を最小限にし、出力は float32 を基本とする。
    """
    x = _as_float01_bchw(x)

    if cfg.impl not in ("simclrv1", "simclrv2"):
        raise ValueError(f"Unknown cfg.impl: {cfg.impl}")

    if cfg.crop:
        x = random_resized_crop_batch(x, out_h=cfg.out_h, out_w=cfg.out_w, p=1.0)
    if cfg.flip:
        x = random_hflip_batch(x, p=0.5)
    if cfg.color_distort:
        x = random_color_distort_batch(x, strength=cfg.color_jitter_strength, impl=cfg.impl, p=1.0)
    if cfg.blur:
        x = gaussian_blur_batch(
            x,
            kernel_size=max(1, cfg.out_h // 10),
            sigma_min=0.1,
            sigma_max=2.0,
            p=cfg.blur_prob,
        )

    return x.clamp(0.0, 1.0)


def preprocess_for_eval_batch(x: Tensor, cfg: SimCLRAugConfig, *, crop: bool = True) -> Tensor:
    """評価時の前処理（バッチ版）。

    学習時のような確率的拡張は行わず、指定解像度へ整形する。
    crop=True の場合は center crop + resize、crop=False の場合は単純な resize を行う。

    Parameters
    ----------
    x : torch.Tensor
        入力画像。uint8 (0..255) または float を許容する。
        shape は (B, 3, H, W)。
    cfg : SimCLRAugConfig
        前処理設定（out_h/out_w を使用）。
    crop : bool, default=True
        True の場合は center crop + resize、False の場合は resize のみ。

    Returns
    -------
    torch.Tensor
        整形後の画像。shape は (B, 3, cfg.out_h, cfg.out_w)。
        値域は [0,1] にクリップされる。

    Raises
    ------
    TypeError
        x が torch.Tensor でない場合。
    ValueError
        x が 4 次元でない場合、または C!=3 の場合（内部の正規化関数で検証される）。

    Notes
    -----
    - center crop の crop_proportion は固定値（CROP_PROPORTION）を用いる。
    - resize は bicubic を用い、align_corners=False で実行する。
    """
    x = _as_float01_bchw(x)
    if crop:
        return _center_crop_and_resize_batch(x, cfg.out_h, cfg.out_w, CROP_PROPORTION)
    return F.interpolate(x, size=(cfg.out_h, cfg.out_w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)