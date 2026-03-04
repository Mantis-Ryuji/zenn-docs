from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from timm.models.resnet import Bottleneck
from torch import Tensor

StemMode: TypeAlias = Literal["auto", "imagenet", "cifar"]


@dataclass(frozen=True)
class ResNet50EncoderConfig:
    """ResNet-50 Encoder の設定。

    SimCLR 等で用いる「分類ヘッドを除いた ResNet-50 の特徴抽出器」を構成するための
    ハイパーパラメータ（入力形状、stem 形状、BatchNorm 設定）を保持する。

    本設定は **出力特徴次元を 2048（layer4 後の global average pooling）** とする
    ResNet-50 の標準構成に対応する。

    Parameters
    ----------
    input_size : tuple[int, int, int], default=(3, 256, 256)
        入力画像サイズ (C, H, W)。
        本実装は RGB 入力を前提とするため C=3 を要求する。

        allow_any_spatial=False の場合、forward で (H, W) の一致を厳格に検証する。
    stem : {"auto", "imagenet", "cifar"}, default="auto"
        Stem の種別。

        - "imagenet": 7x7 stride2 + maxpool（ImageNet 標準 ResNet stem）
        - "cifar":    3x3 stride1 + maxpool なし（小解像度向け）
        - "auto":     min(H, W) <= 64 なら "cifar"、それ以外は "imagenet"
    allow_any_spatial : bool, default=False
        True の場合、(H, W) の厳格チェックを無効化し、(C=3) のみ検証する
        （可変解像度入力を許容する）。
    eps : float, default=1e-5
        BatchNorm2d の eps（分散に加える微小値）。数値安定性のため正値を要求する。
    bn_momentum : float, default=0.1
        BatchNorm2d の momentum（running mean/var の更新係数）。
        通常は 0 < momentum < 1 を推奨する。

    Notes
    -----
    - 出力特徴次元は ResNet-50 の最終段（layer4）後の global average pooling のため 2048。
    - CIFAR-10 等の小解像度では input_size=(3, 32, 32), stem="auto" で "cifar" stem が選択される。
    - allow_any_spatial=True を用いる場合でも、stem と各 stage の stride により
      極端に小さい入力では空間次元が潰れる可能性があるため、実運用では入力解像度の下限を別途設けること。
    """
    input_size: tuple[int, int, int] = (3, 256, 256)
    stem: StemMode = "auto"
    allow_any_spatial: bool = False
    eps: float = 1e-5
    bn_momentum: float = 0.1


class ResNet50Encoder(nn.Module):
    """ResNet-50 Encoder（timm の Bottleneck を用いた特徴抽出器）。

    分類器（fc）を持たず、最終 stage（layer4）後の global average pooling により
    画像を固定次元ベクトルへ写像する。SimCLR では encoder の出力に projection head を
    接続して対照学習を行うため、本クラスは「encoder 部分のみ」を提供する。

    ResNet-50 の段構成は [3, 4, 6, 3]（stage1..4 の Bottleneck ブロック数）であり、
    timm の `Bottleneck` ブロックを用いて手動で組み立てる。

    Attributes
    ----------
    cfg : ResNet50EncoderConfig
        設定。
    out_dim : int
        出力特徴次元（常に 2048）。

    Notes
    -----
    - 入力は (B, C, H, W) の 4 次元テンソルを想定し、C=3 を要求する。
    - stem="imagenet" は stride=2 + maxpool により初段で強くダウンサンプリングするため、
      小解像度（例: 32x32）では情報落ちが大きい。小解像度では stem="cifar" が適する。
    - BatchNorm を用いる。小バッチでは統計が不安定になり得るため、必要に応じて
      (i) バッチを大きくする / (ii) SyncBN / (iii) BN の凍結等を検討する。
    - 本クラスは重みの一部のみ（Conv）に Kaiming 初期化を適用し、BN 等は PyTorch の
      デフォルト初期化に委ねる（`_init_weights()` 参照）。
    """
    out_dim: int = 2048

    def __init__(self, cfg: ResNet50EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResNet50EncoderConfig()

        c, h, w = self.cfg.input_size
        if c != 3:
            raise ValueError(f"ResNet50Encoder は RGB (C=3) 前提です: got C={c}")
        if h < 8 or w < 8:
            raise ValueError(f"(H,W) は十分大きい必要があります: got (H,W)=({h},{w})")
        if self.cfg.eps <= 0.0:
            raise ValueError(f"eps は正である必要があります: got {self.cfg.eps}")
        if not (0.0 < self.cfg.bn_momentum < 1.0):
            raise ValueError(f"bn_momentum は (0,1) が推奨です: got {self.cfg.bn_momentum}")

        stem_mode = self._resolve_stem_mode(stem=self.cfg.stem, h=h, w=w)

        # ---- Stem ----
        if stem_mode == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1 = self._norm_layer(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = self._norm_layer(64)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()

        # ---- Stages (ResNet-50): [3, 4, 6, 3] ----
        self._inplanes = 64
        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()

    @staticmethod
    def _resolve_stem_mode(*, stem: StemMode, h: int, w: int) -> Literal["imagenet", "cifar"]:
        if stem == "imagenet":
            return "imagenet"
        if stem == "cifar":
            return "cifar"
        if stem == "auto":
            return "cifar" if min(h, w) <= 64 else "imagenet"
        # Literal 的には到達不能だが、fail-fast のため明示
        raise ValueError(f"未サポートの stem: {stem}")

    def _norm_layer(self, ch: int, **_: object) -> nn.Module:
        if int(ch) < 1:
            raise ValueError(f"ch は 1 以上である必要があります: got {ch}")
        return nn.BatchNorm2d(
            int(ch),
            eps=float(self.cfg.eps),
            momentum=float(self.cfg.bn_momentum),
            affine=True,
            track_running_stats=True,
        )

    def _make_layer(self, *, planes: int, blocks: int, stride: int) -> nn.Sequential:
        if planes < 1:
            raise ValueError(f"planes は 1 以上である必要があります: got {planes}")
        if blocks < 1:
            raise ValueError(f"blocks は 1 以上である必要があります: got {blocks}")
        if stride not in (1, 2):
            raise ValueError(f"stride は 1 または 2 を想定します: got {stride}")

        outplanes = planes * Bottleneck.expansion

        downsample: nn.Module | None = None
        if stride != 1 or self._inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self._inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(outplanes),
            )

        layers: list[nn.Module] = []
        layers.append(
            Bottleneck(
                self._inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                act_layer=nn.ReLU,
                norm_layer=self._norm_layer, # type: ignore
            )
        )
        self._inplanes = outplanes

        for _ in range(blocks - 1):
            layers.append(
                Bottleneck(
                    self._inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    act_layer=nn.ReLU,
                    norm_layer=self._norm_layer, # type: ignore
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """前向き計算（特徴抽出）。

        入力画像を stem → stage1..4 → global average pooling に通し、固定次元ベクトルを返す。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル。shape は (B, C, H, W)。
            dtype は浮動小数点（例: float32 / bfloat16 / float16）を想定する。

        Returns
        -------
        torch.Tensor
            特徴ベクトル。shape は (B, 2048)。

        Raises
        ------
        TypeError
            x が torch.Tensor でない場合。
        ValueError
            - x が 4 次元でない場合
            - C が `cfg.input_size[0]` と一致しない場合
            - allow_any_spatial=False で (H, W) が `cfg.input_size[1:3]` と一致しない場合

        Notes
        -----
        - allow_any_spatial=True のときは (H, W) の検証を行わず、可変解像度入力を許容する。
        ただし stem と各 stage の stride により、極端に小さい入力では空間次元が潰れる可能性があるため、
        実運用では入力解像度の下限を別途管理すること。
        - 本メソッドは device/dtype の移動を行わない（呼び出し側で `to(device)` 等を行う）。
        - 出力は global average pooling 後に `torch.flatten(x, 1)` で (B, 2048) に整形される。
        """
        if not isinstance(x, torch.Tensor): # type: ignore
            raise TypeError(f"x must be torch.Tensor. got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(f"x は 4 次元 (B,C,H,W) が必要です: got shape={tuple(x.shape)}")

        c, h, w = self.cfg.input_size
        if x.shape[1] != c:
            raise ValueError(f"入力チャネル不一致: expected C={c}, got C={x.shape[1]}")
        if not self.cfg.allow_any_spatial:
            if (x.shape[2], x.shape[3]) != (h, w):
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
        x = torch.flatten(x, 1)
        return x