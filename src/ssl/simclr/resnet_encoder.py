from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from timm.models.resnet import Bottleneck
from torch import Tensor

StemMode: TypeAlias = Literal["auto", "imagenet", "cifar"]

class LayerNorm2d(nn.Module):
    """2次元特徴マップ（NCHW）向けの LayerNorm。

    Conv2d 出力のような NCHW テンソルに対し、各空間位置 (h, w) ごとに
    **チャネル次元 C のみ**を正規化する LayerNorm を提供する。

    Notes
    -----
    - PyTorch の ``nn.LayerNorm(normalized_shape=C)`` は入力の「最後の次元」を正規化する。
      そのため NCHW のままでは適用できない（最後の次元が W になってしまう）。
      本クラスでは ``permute`` により NCHW -> NHWC に変換し、LN を適用してから元に戻す。

    Shapes
    ------
    - Input:  x # (B, C, H, W)
    - Output: y # (B, C, H, W)

    Parameters
    ----------
    num_channels : int
        正規化するチャネル数 C。
    eps : float, default=1e-5
        数値安定化項。
    elementwise_affine : bool, default=True
        学習可能な affine（scale/bias）を持つかどうか。
    device : torch.device | str | int | None, default=None
        パラメータを配置するデバイス。None の場合は PyTorch のデフォルトに従う。
    dtype : torch.dtype | None, default=None
        パラメータの dtype。None の場合は PyTorch のデフォルトに従う。

    Raises
    ------
    ValueError
        num_channels < 1、eps <= 0、または forward 入力が 4 次元でない場合。
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if int(num_channels) < 1:
            raise ValueError(f"num_channels は 1 以上である必要があります: got {num_channels}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {eps}")

        self.ln = nn.LayerNorm(
            num_channels,
            eps=eps,
            elementwise_affine=bool(elementwise_affine),
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """LayerNorm を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル。x # (B, C, H, W)

        Returns
        -------
        torch.Tensor
            正規化後テンソル。y # (B, C, H, W)

        Raises
        ------
        ValueError
            x が 4 次元でない場合。
        """
        if x.ndim != 4:
            raise ValueError(f"x は 4 次元 (B,C,H,W) である必要があります: got shape={tuple(x.shape)}")

        # x # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # x # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


@dataclass(frozen=True)
class ResNet50EncoderConfig:
    """ResNet-50 Encoder 設定（SimCLR 用に encoder のみ使う想定）。

    Parameters
    ----------
    input_size : tuple[int, int, int], default=(3, 256, 256)
        入力画像サイズ (C,H,W)。
        allow_any_spatial=False の場合、forward で (C,H,W) 一致を検証する。
    stem : {"auto","imagenet","cifar"}, default="auto"
        Stem の種別。
        - "imagenet": 7x7 stride2 + maxpool stride2（標準 ResNet）
        - "cifar":    3x3 stride1 + maxpool なし（小解像度向け）
        - "auto":     min(H,W) <= 64 なら "cifar"、それ以外は "imagenet"
    allow_any_spatial : bool, default=False
        True の場合、(H,W) の厳格チェックを無効化（C=3 のみ検証）。
    eps : float, default=1e-5
        LayerNorm の eps。

    Notes
    -----
    - デフォルトは (3,256,256) + stem="auto" なので実質 "imagenet" stem となり、
      既存の「256 前提」挙動を維持する。
    - CIFAR-10 等で使う場合は input_size=(3,32,32), stem="auto" で "cifar" stem が選ばれる。
    """

    input_size: tuple[int, int, int] = (3, 256, 256)
    stem: StemMode = "auto"
    allow_any_spatial: bool = False
    eps: float = 1e-5


class ResNet50Encoder(nn.Module):
    """ResNet-50 Encoder（timm Bottleneck 使用）。

    目的
    ----
    入力画像
        x ∈ R^{B×3×H×W}
    を特徴ベクトル
        z ∈ R^{B×2048}
    に写像する encoder を提供する。

    実装方針
    --------
    - 残差ブロックは timm.models.resnet.Bottleneck を使用する。:contentReference[oaicite:3]{index=3}
    - 正規化は LayerNorm2d を用い、timm Bottleneck の norm_layer 引数に渡す。
      timm Bottleneck は norm_layer(ch) を呼び出す設計であるため、LayerNorm2d はその契約を満たす。:contentReference[oaicite:4]{index=4}
    - 出力は global average pooling の後に LayerNorm(2048)。

    Parameters
    ----------
    cfg : ResNet50EncoderConfig | None, default=None
        Encoder 設定。

    Returns
    -------
    z : torch.Tensor
        shape=(B, 2048) の特徴ベクトル。

    Raises
    ------
    TypeError
        入力が torch.Tensor でない場合。
    ValueError
        入力 shape が不正、または cfg の前提に反する場合。
    """

    def __init__(self, cfg: ResNet50EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResNet50EncoderConfig()

        c, h, w = self.cfg.input_size
        if int(c) != 3:
            raise ValueError(f"ResNet50Encoder は RGB (C=3) 前提です: got C={c}")
        if int(h) < 8 or int(w) < 8:
            raise ValueError(f"(H,W) は十分大きい必要があります: got (H,W)=({h},{w})")
        if float(self.cfg.eps) <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {self.cfg.eps}")

        # stem mode
        if self.cfg.stem == "imagenet":
            stem_mode: Literal["imagenet", "cifar"] = "imagenet"
        elif self.cfg.stem == "cifar":
            stem_mode = "cifar"
        elif self.cfg.stem == "auto":
            stem_mode = "cifar" if min(int(h), int(w)) <= 64 else "imagenet"
        else:
            raise ValueError(f"未サポートの stem: {self.cfg.stem}")

        # norm/act factories for timm Bottleneck
        # Bottleneck(..., act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d) のように渡される設計。
        self._act_layer = nn.ReLU
        def _norm_layer(ch: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None, **_: object) -> nn.Module:
            return LayerNorm2d(int(ch), eps=self.cfg.eps, device=device, dtype=dtype)
        
        self._norm_layer = _norm_layer

        # ---- Stem ----
        if stem_mode == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1 = self._norm_layer(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # CIFAR 向け：序盤で潰しすぎない
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = self._norm_layer(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()

        # ---- Stages (ResNet-50): [3,4,6,3] ----
        self._inplanes = 64
        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_norm = nn.LayerNorm(2048, eps=self.cfg.eps)

    def _make_layer(self, *, planes: int, blocks: int, stride: int) -> nn.Sequential:
        if int(planes) < 1:
            raise ValueError(f"planes は 1 以上である必要があります: got {planes}")
        if int(blocks) < 1:
            raise ValueError(f"blocks は 1 以上である必要があります: got {blocks}")
        if int(stride) not in (1, 2):
            raise ValueError(f"stride は 1 または 2 を想定します: got {stride}")

        outplanes = int(planes) * Bottleneck.expansion  # timm Bottleneck.expansion = 4

        downsample: nn.Module | None = None
        if stride != 1 or self._inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self._inplanes, outplanes, kernel_size=1, stride=int(stride), bias=False),
                self._norm_layer(outplanes),
            )

        layers: list[nn.Module] = []
        layers.append(
            Bottleneck(
                self._inplanes,
                int(planes),
                stride=int(stride),
                downsample=downsample,
                act_layer=self._act_layer,
                norm_layer=self._norm_layer, # type: ignore
            )
        )
        self._inplanes = outplanes

        for _ in range(int(blocks) - 1):
            layers.append(
                Bottleneck(
                    self._inplanes,
                    int(planes),
                    stride=1,
                    downsample=None,
                    act_layer=self._act_layer,
                    norm_layer=self._norm_layer, # type: ignore
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):  # type: ignore
            raise TypeError(f"x must be torch.Tensor. got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"x は 4 次元 (B,C,H,W) が必要です: got shape={tuple(x.shape)}")

        c, h, w = self.cfg.input_size
        if x.shape[1] != int(c):
            raise ValueError(f"入力チャネル不一致: expected C={c}, got C={x.shape[1]}")
        if not self.cfg.allow_any_spatial:
            if (x.shape[2], x.shape[3]) != (int(h), int(w)):
                raise ValueError(
                    f"入力空間サイズ不一致: expected (H,W)=({h},{w}), got (H,W)=({x.shape[2]},{x.shape[3]})"
                )

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.out_norm(x)
        return x