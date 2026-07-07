---
title: "個人的まとめ：2026年度で精読した論文リスト"
emoji: "📑"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AI","深層学習"]
published: false
---

## はじめに

AIが発達し、簡単に論文のまとめを出せる中で、わかった気になるような状態になることが多い気がしたので、新しい勉強法として、Zenn を外部検証装置として使ってきた。まあパワポで発表するよりかは劣るが、一人で論文を読むときよりかはより内容を精査するので効果があるような気もする。またモチベ維持につながる。

---

## 1. 汎化

汎化は機械学習における OS だと思っている。モデルが変わろうと変わらないので、汎化を考える力を身に着けたく、勉強中。

### 1.1 データ拡張

@[card](https://arxiv.org/abs/1710.09412)
> 訓練データ同士を線形補間することで新たな学習サンプルを作る単純なデータ拡張手法。
> 記事：[「混ぜる」データ拡張は本当に有効なのか？ mixup が示した汎化と頑健性](https://zenn.dev/mantis_ryuji/articles/9f9aae80ae30c3)

---

### 1.2 損失地形

@[card](https://arxiv.org/abs/1609.04836)
> 大バッチ学習では、近傍摂動に対して脆い sharp minima に到達しやすいという仮説。
> 記事：[大バッチ学習はなぜ汎化しにくいのか：Sharp Minima 論文から見る最適化と汎化の関係](https://zenn.dev/mantis_ryuji/articles/3a6c3b210ffdeb)

@[card](https://arxiv.org/abs/1703.04933)
> 上の sharp minima 仮説に対する反論。
> 記事：[Sharp Minima は本当に汎化を説明するのか：大バッチ学習と再パラメータ化からの反論](https://zenn.dev/mantis_ryuji/articles/0f7b2bb4dd8696)

@[card](https://arxiv.org/abs/1802.10026)
> 低損失領域の観察と、それを活かした高速アンサンブル手法。
> 記事：[Fast Geometric Ensembling 論文解説：低損失経路から高速アンサンブルへ](https://zenn.dev/mantis_ryuji/articles/27dbe201808952)

---

### 1.3 ドメイン汎化手法

@[card](https://arxiv.org/abs/2007.01434)
> ドメイン汎化を研究するなら格好いい数式を並べる前に評価設計をしっかりしようという論文。
> 記事：[DomainBed 論文解説: ドメイン汎化研究における評価設計の問題](https://zenn.dev/mantis_ryuji/articles/d3f865ccb42028)

@[card](https://arxiv.org/abs/1911.08731)
> 過剰パラメータ化モデルの最悪グループ性能を明示的に改善する手法（グループの作成には、ラベルのほかに疑似属性が必要なため、ラベル付けがやや面倒）。
> 記事：[Group DRO 論文解説：過剰パラメータ化モデルの最悪グループ汎化](https://zenn.dev/mantis_ryuji/articles/0faceb3e34e9ab)

@[card](https://arxiv.org/abs/2107.09044)
> GroupDRO に必要なグループアノテーションのコストを緩和する手法（検証セットには必要）。
> 記事：[Just Train Twice 論文解説：失敗例を重く見るだけで group robustness を改善するシンプルな手法](https://zenn.dev/mantis_ryuji/articles/ccde9418c85859) 

@[card](https://arxiv.org/abs/2302.11861)
> ドメイン知識を用いて疑似相関を特定し、それを意図的に壊すデータ拡張を設計する手法。
> 記事：[Out-of-Domain Robustness via Targeted Augmentations 論文解説](https://zenn.dev/mantis_ryuji/articles/d0120266f25ceb) 

ドメイン汎化手法の論文は、流し読みも含めると有名どころは大体ざっと読んだが、難しい数式をこねくり回すより、評価設計・データ設計・アノテーション設計をちゃんとやることの方が重要なのではないか、と感じている。

結局、ドメイン汎化の実践では、数式で目的関数を工夫するだけでは足りず、どの分布差を評価し、どの失敗を重く見て、どの相関を壊すかという設計が本質になりやすいのだろう。そうした泥臭い設計判断が、最終的な OOD 性能を大きく左右する。

その意味で、これらの論文を読むほど、ドメイン汎化は「かっこいい汎用アルゴリズムを探す問題」というより、「データと評価をどう設計するか」という問題なのだと感じた。

---

### 1.4 その他

@[card](https://arxiv.org/abs/1611.03530)
> 従来の汎化を説明する指標では、もはやニューラルネットワークの汎化を十分に捉えられないという論文。
> 記事：[ニューラルネットワークは、なぜ記憶できるのに汎化するのか？](https://zenn.dev/mantis_ryuji/articles/2536165a410e79) 

@[card](https://arxiv.org/abs/2205.10343)
> Grokking を表現学習の遅延として捉えた論文。
> 記事：[ニューラルネットワークは FizzBuzz を「理解」できるのか――未知の桁数への外挿実験](https://zenn.dev/mantis_ryuji/articles/fizz-buzz-exp) 

@[card](https://arxiv.org/abs/2301.05217)
> Grokking を記憶回路から汎化回路への移行として捉えた論文。
> 記事：[ニューラルネットワークは FizzBuzz を「理解」できるのか――未知の桁数への外挿実験](https://zenn.dev/mantis_ryuji/articles/fizz-buzz-exp) 

---

## 2. 自己教師あり学習

ラベル無しデータを有効活用できればいいなと思い、勉強中。

@[card](https://arxiv.org/abs/2002.05709)
> 同一サンプルから生成した 2 つの拡張ビューの表現を一致させるというシンプルな手法。
> 記事：[SimCLR 論文解説 ＆ PyTorch 実装（Contrastive Learning / 自己教師あり学習）](https://zenn.dev/mantis_ryuji/articles/8018d3c45c4290) 

@[card](https://arxiv.org/abs/2006.07733)
> 非対称性と安定化によって自己予測を成立させ、表現学習の新しい成立条件を示した論文。
> 記事：[BYOLを理解する：負例なし学習と崩壊回避のメカニズム](https://zenn.dev/mantis_ryuji/articles/c8076edeeae3b0) 

@[card](https://arxiv.org/abs/2105.04906)
> 表現崩壊の回避を、アーキテクチャ上の特殊な工夫ではなく、目的関数側の明示的な正則化によって実現する手法。
> 記事：[VICReg: 自己教師あり学習における崩壊回避の明示的設計](https://zenn.dev/mantis_ryuji/articles/307a04e64e06a2) 

@[card](https://arxiv.org/abs/2111.06377)
> Masked Reconstruction を Vision に適した表現学習課題へ作り直した論文。
> 記事：[Masked Autoencoders (MAE) 論文解説 ― なぜ高マスク率と非対称設計が効くのか](https://zenn.dev/mantis_ryuji/articles/67e096d62c9265) 

@[card](https://arxiv.org/abs/2301.00808)
> ConvNet の構造に合わせて masked autoencoder を設計し直した論文。
> 記事：[ConvNeXt V2 論文解説：CNN のための Masked Autoencoder](https://zenn.dev/mantis_ryuji/articles/9628b8eef173d4)

ここら辺は何を持ち帰ればいいのだろうか。面白いんだが汎化とは違って手法そのものは陳腐化しやすい。思想を学べたらいいのかなあ。
それにしても MIM 系が安定択すぎる。

---

## 3. 個人的面白かったランキング


---

## 4. さいごに