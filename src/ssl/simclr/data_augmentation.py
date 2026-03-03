from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch import Tensor

# ImageNet の標準
CROP_PROPORTION: float = 0.875


@dataclass(frozen=True)
class SimCLRAugConfig:
    """SimCLR 画像前処理（バッチ版）の設定。

    Parameters
    ----------
    out_h : int=256
        出力画像の高さ。
    out_w : int=256
        出力画像の幅。
    color_jitter_strength : float, default=1.0
        TF版の FLAGS.color_jitter_strength 相当。color_jitter の強度。
    impl : str, default="simclrv2"
        brightness 実装。
        - "simclrv2": 乗算（factor を掛ける）
        - "simclrv1": 加算（delta を足す）
    crop : bool, default=True
        学習時に RandomResizedCrop を行うか。
    flip : bool, default=True
        学習時に左右反転を行うか。
    color_distort : bool, default=True
        学習時に color jitter + grayscale を行うか。
    blur : bool, default=True
        学習時に Gaussian blur を行うか（SimCLR 標準レシピ）。
    blur_prob : float, default=0.5
        blur の適用確率（各サンプル独立）。
    """

    out_h: int = 256
    out_w: int = 256
    color_jitter_strength: float = 1.0
    impl: str = "simclrv2"
    crop: bool = True
    flip: bool = True
    color_distort: bool = True
    blur: bool = True
    blur_prob: float = 0.5


def _as_float01_bchw(x: Tensor) -> Tensor:
    """入力を float32 / [0,1] / (B,C,H,W) に正規化する（Fail Fast）。"""
    if not isinstance(x, Tensor): # type: ignore
        raise TypeError(f"x must be torch.Tensor. got {type(x)}")
    if x.ndim != 4:
        raise ValueError(f"x must be 4D (B,C,H,W). got shape={tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"expected C=3. got C={x.shape[1]}")

    if x.dtype == torch.uint8:
        x = x.float().div(255.0)
    else:
        x = x.float()

    return x.clamp_(0.0, 1.0)


def _rand_bool_mask(bsz: int, p: float, *, device: torch.device) -> Tensor:
    """Bernoulli mask. mask # (B,) bool"""
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be in [0,1]. got {p}")
    return torch.rand((bsz,), device=device) < float(p)


def _apply_where(mask_b: Tensor, x_new: Tensor, x_old: Tensor) -> Tensor:
    """mask_b # (B,) を (B,1,1,1) に拡張して select."""
    if mask_b.ndim != 1:
        raise ValueError(f"mask must be 1D (B,). got shape={tuple(mask_b.shape)}")
    m = mask_b[:, None, None, None]
    return torch.where(m, x_new, x_old)


# -------------------------
# Color jitter primitives (batch-safe)
# -------------------------
def _random_brightness(x: Tensor, max_delta: float, *, impl: str) -> Tensor:
    """brightness（SimCLRv2: 乗算 / SimCLRv1: 加算）。x # (B,3,H,W)"""
    bsz = x.shape[0]
    device = x.device
    if float(max_delta) <= 0.0:
        return x

    if impl == "simclrv2":
        lo = max(1.0 - float(max_delta), 0.0)
        hi = 1.0 + float(max_delta)
        factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(lo, hi)
        return (x * factor).clamp(0.0, 1.0)
    if impl == "simclrv1":
        delta = torch.empty((bsz, 1, 1, 1), device=device).uniform_(-float(max_delta), float(max_delta))
        return (x + delta).clamp(0.0, 1.0)

    raise ValueError(f"Unknown impl for brightness: {impl}")


def _random_contrast(x: Tensor, max_delta: float) -> Tensor:
    """contrast をランダムに（バッチ対応）。x # (B,3,H,W)"""
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(1.0 - float(max_delta), 1.0 + float(max_delta))
    mean = x.mean(dim=(2, 3), keepdim=True)  # (B,3,1,1)
    y = (x - mean) * factor + mean
    return y.clamp(0.0, 1.0)


def _random_saturation(x: Tensor, max_delta: float) -> Tensor:
    """saturation をランダムに（バッチ対応）。x # (B,3,H,W)"""
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    factor = torch.empty((bsz, 1, 1, 1), device=device).uniform_(1.0 - float(max_delta), 1.0 + float(max_delta))
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])  # (B,1,H,W)
    gray3 = gray.expand(-1, 3, -1, -1)
    y = x * factor + gray3 * (1.0 - factor)
    return y.clamp(0.0, 1.0)


def _rgb_to_hsv(x: Tensor) -> Tensor:
    """x # (B,3,H,W) in [0,1] -> hsv # (B,3,H,W)"""
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
    h = (h / 6.0) % 1.0  # [0,1)

    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: Tensor) -> Tensor:
    """hsv # (B,3,H,W) -> rgb # (B,3,H,W)"""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h6 = (h * 6.0) % 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i.float()

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))

    return torch.stack([r, g, b], dim=1)


def _random_hue(x: Tensor, max_delta: float) -> Tensor:
    """hue をランダムに（バッチ対応）。x # (B,3,H,W)"""
    if float(max_delta) <= 0.0:
        return x
    bsz = x.shape[0]
    device = x.device
    delta = torch.empty((bsz, 1, 1), device=device).uniform_(-float(max_delta), float(max_delta))
    hsv = _rgb_to_hsv(x)
    h = (hsv[:, 0] + delta).remainder(1.0)
    hsv = torch.stack([h, hsv[:, 1], hsv[:, 2]], dim=1)
    y = _hsv_to_rgb(hsv)
    return y.clamp(0.0, 1.0)


def _to_grayscale_keep3(x: Tensor) -> Tensor:
    """RGB->Gray（3ch維持）。x # (B,3,H,W)"""
    return TVF.rgb_to_grayscale(x, num_output_channels=3)


def color_jitter_batch(
    x: Tensor,
    *,
    strength: float,
    random_order: bool = True,
    impl: str = "simclrv2",
) -> Tensor:
    """SimCLR の color jitter（バッチ版、ベクトル化）。

    Notes
    -----
    random_order=True の場合、**サンプルごとに順序をシャッフル**する。
    バッチ方向の分岐を避けるため、各ステップで 4 変換候補を全て計算し、
    perm に従って mask で選択して合成する（ステップ数 4 の固定ループのみ）。

    Parameters
    ----------
    x : torch.Tensor
        入力。x # (B, 3, H, W), range [0,1]
    strength : float
        強度。brightness/contrast/saturation は 0.8*strength、hue は 0.2*strength。
    random_order : bool, default=True
        jitter の順序をランダムにするか。
    impl : str, default="simclrv2"
        brightness の実装。

    Returns
    -------
    torch.Tensor
        出力。y # (B, 3, H, W)
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

    if not random_order:
        y = x
        for fn in (f_b, f_c, f_s, f_h):
            y = fn(y)
        return y.clamp(0.0, 1.0)

    bsz = x.shape[0]
    device = x.device

    # perm # (B,4): 各行が [0,1,2,3] のランダム順列
    perm = torch.stack([torch.randperm(4, device=device) for _ in range(bsz)], dim=0)

    y = x
    for t in range(4):
        # 同一入力 y から 4候補を並列生成
        y0 = f_b(y)
        y1 = f_c(y)
        y2 = f_s(y)
        y3 = f_h(y)

        idx = perm[:, t]  # (B,)

        y = _apply_where(idx == 0, y0, y)
        y = _apply_where(idx == 1, y1, y)
        y = _apply_where(idx == 2, y2, y)
        y = _apply_where(idx == 3, y3, y)

        y = y.clamp(0.0, 1.0)

    return y


def random_color_jitter_batch(
    x: Tensor,
    *,
    strength: float,
    impl: str,
    p: float = 1.0,
) -> Tensor:
    """SimCLR の random_color_jitter（バッチ版）。

    TF版相当：
      - 全体 transform を p で適用
      - color_jitter を 0.8 で適用
      - grayscale を 0.2 で適用
    """
    bsz = x.shape[0]
    device = x.device

    m_all = _rand_bool_mask(bsz, p, device=device)
    if not m_all.any().item():
        return x

    y = x

    m_j = _rand_bool_mask(bsz, 0.8, device=device) & m_all
    if m_j.any().item():
        y_j = color_jitter_batch(y, strength=strength, random_order=True, impl=impl)
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
    """TF版 center_crop 相当（バッチ版）。"""
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
    """SimCLR の crop_and_resize 相当（バッチ版・ベクトル化）。

    TF版 sample_distorted_bounding_box(max_attempts=100) の厳密再現は避け、
    1回サンプルでクロップ領域を決める（ベクトル化優先）。
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

    yy = torch.linspace(0, 1, steps=out_h, device=device)[None, :, None]  # (1,OH,1)
    xx = torch.linspace(0, 1, steps=out_w, device=device)[None, None, :]  # (1,1,OW)

    crop_h_f = (crop_h - 1).clamp(min=1).float()[:, None, None]
    crop_w_f = (crop_w - 1).clamp(min=1).float()[:, None, None]
    y_pix = off_y.float()[:, None, None] + yy * crop_h_f
    x_pix = off_x.float()[:, None, None] + xx * crop_w_f

    y_pix = y_pix.expand(b, out_h, out_w)
    x_pix = x_pix.expand(b, out_h, out_w)

    y_norm = (y_pix / max(h - 1, 1)) * 2.0 - 1.0
    x_norm = (x_pix / max(w - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,OH,OW,2)

    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    y = y.clamp(0.0, 1.0)

    return _apply_where(m, y, x)


def random_hflip_batch(x: Tensor, p: float = 0.5) -> Tensor:
    """左右反転（バッチ版）。"""
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
    """Gaussian blur（バッチ版・sigma を各サンプル独立にしてもベクトル化）。

    TF版に合わせて separable depthwise conv（水平→垂直）で実装する。
    各サンプルで sigma が異なるため (B*C) groups の grouped conv で一括処理する。
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

    xs = torch.arange(-radius, radius + 1, device=device).float()[None, :]  # (1,K)
    sig = sigma[:, None].clamp(min=1e-6)
    ker = torch.exp(-(xs**2) / (2.0 * (sig**2)))
    ker = ker / ker.sum(dim=1, keepdim=True)

    inp = x.reshape(1, b * c, h, w)

    ker_h = ker[:, None, None, :].repeat_interleave(c, dim=0)  # (B*C,1,1,K)
    ker_v = ker[:, None, :, None].repeat_interleave(c, dim=0)  # (B*C,1,K,1)

    pad = radius
    out = F.conv2d(inp, ker_h, bias=None, stride=1, padding=(0, pad), groups=b * c)
    out = F.conv2d(out, ker_v, bias=None, stride=1, padding=(pad, 0), groups=b * c)
    out = out.reshape(b, c, h, w).clamp(0.0, 1.0)

    return _apply_where(m, out, x)


# -------------------------
# Public preprocess API (batch)
# -------------------------
def preprocess_for_train_batch(x: Tensor, cfg: SimCLRAugConfig) -> Tensor:
    """学習時の前処理（バッチ版）。"""
    x = _as_float01_bchw(x)

    if cfg.crop:
        x = random_resized_crop_batch(x, out_h=cfg.out_h, out_w=cfg.out_w, p=1.0)

    if cfg.flip:
        x = random_hflip_batch(x, p=0.5)

    if cfg.color_distort:
        x = random_color_jitter_batch(
            x,
            strength=cfg.color_jitter_strength,
            impl=cfg.impl,
            p=1.0,
        )

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
    """評価時の前処理（バッチ版）。"""
    x = _as_float01_bchw(x)

    if crop:
        x = _center_crop_and_resize_batch(x, cfg.out_h, cfg.out_w, CROP_PROPORTION)
    else:
        x = F.interpolate(x, size=(cfg.out_h, cfg.out_w), mode="bicubic", align_corners=False)

    return x.clamp(0.0, 1.0)


def preprocess_image_batch(
    x: Tensor,
    cfg: SimCLRAugConfig,
    *,
    is_training: bool,
    color_distort: bool = True,
    test_crop: bool = True,
) -> Tensor:
    """TF版 preprocess_image 相当（バッチ版）。

    Parameters
    ----------
    x : torch.Tensor
        入力。x # (B, 3, H, W)
    cfg : SimCLRAugConfig
        前処理設定。
    is_training : bool
        学習用かどうか。
    color_distort : bool, default=True
        学習時の color jitter を有効にするか（cfg.color_distort を上書き）。
    test_crop : bool, default=True
        評価時の center crop を行うか。

    Returns
    -------
    torch.Tensor
        前処理済みテンソル。y # (B, 3, out_h, out_w), range [0,1]
    """
    if is_training:
        cfg2 = SimCLRAugConfig(
            out_h=cfg.out_h,
            out_w=cfg.out_w,
            color_jitter_strength=cfg.color_jitter_strength,
            impl=cfg.impl,
            crop=cfg.crop,
            flip=cfg.flip,
            color_distort=bool(color_distort),
            blur=cfg.blur,
            blur_prob=cfg.blur_prob,
        )
        return preprocess_for_train_batch(x, cfg2)

    return preprocess_for_eval_batch(x, cfg, crop=bool(test_crop))