# Hyperparameter Ensembles の図

対象記事は `articles/hyperparameter-ensembles.md`。数理説明用の自作図と、原論文の付属PDFからの変換を再生成する。原資料は変更しない。

## 実行

Python と以下の依存パッケージがある環境で、リポジトリルートから実行する。

```powershell
python -m pip install numpy matplotlib Pillow pypdfium2
python scripts/hyperparameter-ensembles/generate_figures.py --only example
python scripts/hyperparameter-ensembles/generate_figures.py --only paper
```

`--only all`（省略時）で両方を生成する。標準の出力先は `images/zenn/hyperparameter-ensembles/`。同名の生成物は上書きする。比較用に別の出力先へ書くには `--output-dir <directory>` を指定する。

自作図は Python 3.12 / NumPy 2.5.3 / Matplotlib 3.11.2 / Pillow 12.3.0 で確認した。原論文の図の変換には pypdfium2 を使う。

## 自作図の条件

`complementarity.png` は、検証例が2個の人工例である。A = (0.90, 0.30)、B = (0.85, 0.30)、C = (0.40, 0.60) は、各例の正解クラスへの予測確率を並べたもの。二つの座標は別の入力に対応し、和を1にする制約はない。

各候補の NLL は `-mean(log(probabilities))`、2モデルの予測は座標ごとの算術平均から計算する。等高線は `-0.5 * log(q1 * q2)`。実験・学習は行わず、乱数も使わない。本文の数値と候補の順位を出力して確認できる。

## 原論文の図の変換

出典：Wenzel ら, [Hyperparameter Ensembles for Robustness and Uncertainty Quantification](https://arxiv.org/abs/2006.13570), NeurIPS 2020。

入力は `paper/Hyperparameter Ensembles for Robustness and Uncertainty Quantification/figures/`。別の場所を使う場合は、文献のディレクトリを `--paper-dir <directory>` へ渡す。

| 元のPDF | 出力PNG | 原論文の場所 |
| --- | --- | --- |
| `blue_red_figure_init_vs_lambdas.pdf` | `diversity-grid.png` | Figure 2 左 |
| `example_of_lower_upper_lambda_dynamic.pdf` | `learned-ranges.png` | Figure 2 右 |
| `hparam_ens_cifar100_test_acc.pdf` | `cifar100-accuracy.png` | Figure 1 上 |
| `hparam_ens_cifar100_test_ce.pdf` | `cifar100-nll.png` | Figure 1 下 |
| `corruption_cifar10_accuracy.pdf` | `corruption-accuracy.png` | Figure 3 |
| `corruption_cifar10_loss.pdf` | `corruption-nll.png` | Figure 7 下（付録、27ページ） |

標準は200 dpi、RGBのPNG。図の内容を切り替えたり数値を再描画したりせず、各1ページの付属PDFをそのまま画像へ変換する。別のレンダラーで作った画像とはアンチエイリアスなどが異なる場合がある。

書き出し後に全画像を開き、ファイルサイズが各3,000,000バイト以内であることを検査する。解像度や文字の可読性はZennプレビューでも確認する。

記事内の二つの Mermaid 図は、そのフェンス自体が生成ソースである。本文の式・記号と対応させ、原論文の図の転載とは区別する。
