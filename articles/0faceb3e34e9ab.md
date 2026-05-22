---
title: "Group DRO 論文解説：過剰パラメータ化モデルの最悪グループ汎化"
emoji: "🐡"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AI", "深層学習"]
published: false
---

## Group DRO 論文解説：過剰パラメータ化モデルの最悪グループ汎化

機械学習モデルは通常、訓練データ上の平均損失を小さくするように学習される。この目的は、独立同分布（independent and identically distributed; i.i.d.）のテストセット上で高い平均精度を得るという標準的な評価設定と対応している。すなわち、多くの機械学習モデルは、訓練分布とテスト分布が同じであるという前提のもとで、平均的に良い予測を行うように設計されている。

しかし、平均精度が高いことは、すべてのデータグループで性能が高いことを意味しない。データ中にまれで非典型的なグループが存在する場合、モデルは平均的には高精度であっても、そのグループに対して一貫して失敗することがある。この問題は、モデルが **疑似相関** (spurious correlation) に依存している場合に顕著である。

疑似相関とは、訓練データ中の多くの例ではラベルと相関しているが、予測対象の本質的な特徴ではない属性である。たとえば、自然言語推論では、`never` のような否定語の存在が「矛盾」というラベルと強く相関している場合がある。このとき、モデルは文の意味的関係そのものではなく、否定語の有無を手がかりとして予測する可能性がある。そのようなモデルは、i.i.d. なテストセット上では高い平均精度を示し得る。しかし、否定語を含まない矛盾文のように、その疑似相関が成り立たないグループでは大きく失敗する。

本論文が扱う問題設定は、このような疑似相関に起因する **group shift** である。訓練時には、ラベル $y$ と疑似属性 $a$ が相関している。しかし、テスト時にはその相関が成り立つとは限らない。Figure 1 は、Waterbirds、CelebA、MultiNLI の3つのタスクにおいて、この構造を具体例として示している。

![](https://static.zenn.studio/user-upload/c4f8b6db1012-20260522.png)
*Figure 1. 本論文で扱うデータセットにおける代表的な訓練例とテスト例。訓練時に存在するラベル $y$ と疑似属性 $a$ の相関は、テスト時には成り立たない。*

このような問題に対して、自然な方針は、平均損失ではなく、グループごとの最悪ケース損失を小さくすることである。そこで本論文では、Distributionally Robust Optimization（DRO）の一種である Group DRO を用いる。Group DRO は、事前に定義されたグループ集合に対して、最悪グループの訓練損失を最小化するようにモデルを学習する。

ただし、本論文の主張は「Group DRO を使えば最悪グループ性能が改善する」という単純なものではない。むしろ重要なのは、過剰パラメータ化されたニューラルネットワークに naive に Group DRO を適用しても失敗し得る、という観察である。過剰パラメータ化モデルは訓練データを完全にフィットできる。そのため、平均訓練損失が消失するモデルは、すでに最悪ケース訓練損失も消失させている。言い換えると、訓練データ上では ERM と Group DRO の差が見えにくくなる。

このとき問題となるのは、最悪グループの訓練損失ではない。モデルは訓練時には、最悪グループでさえほぼ完全に分類できている。それにもかかわらず、テスト時の最悪グループ性能は低い。したがって、問題の本体は worst-group training loss ではなく、**グループごとの汎化ギャップ** にある。

本論文は、この点から正則化の重要性を示す。過剰パラメータ化領域では、平均汎化の観点では強い正則化が不要に見える場合がある。しかし、最悪グループ汎化を考えると、通常より強い $\ell_2$ ペナルティや早期終了によって、グループごとの汎化ギャップを制御することが重要になる。実際、本論文では、正則化を強化した Group DRO によって、平均精度を高く維持しながら、最悪グループ精度が大きく改善することが示される。

本記事では、まず ERM、DRO、Group DRO の定式化を整理する。次に、Group DRO が見ているリスクを明確にしたうえで、過剰パラメータ化モデルにおいて naive Group DRO が失敗する理由を説明する。その後、正則化、group adjustment、importance weighting との違い、最適化アルゴリズム、実験結果を順に見ていく。

@[card](https://arxiv.org/pdf/1911.08731)
@[card](https://github.com/kohpangwei/group_DRO)

---

## 1. Setup：ERM, DRO, Group DRO

入力特徴量 $x\in\mathcal{X}$ からラベル $y\in\mathcal{Y}$ を予測する問題を考える。モデル族を $\Theta$、損失関数を

$$
\ell:\Theta\times(\mathcal{X}\times\mathcal{Y})\rightarrow\mathbb{R}_{+}
$$

とする。ここで、$\ell$ は非負の損失関数である。また、訓練データはある分布 $P$ から得られるとする。

標準的な教師あり学習では、同じ分布 $P$ のもとで期待損失

$$
\mathbb{E}_{P}[\ell(\theta;(x,y))]
$$

を小さくするモデル $\theta\in\Theta$ を求める。この目的に対する標準的な訓練手続きが、経験リスク最小化（empirical risk minimization; ERM）である。訓練データ上の経験分布を $\hat{P}$ とすると、ERM は次のように定義される。

$$
\hat{\theta}_{\mathrm{ERM}}:=\underset{\theta\in\Theta}{\argmin}\;\mathbb{E}_{(x,y)\sim \hat{P}}[\ell(\theta;(x,y))].\tag{1}
$$

ERM は、訓練データ上の平均損失を最小化する方法である。したがって、訓練分布とテスト分布が同じであり、評価指標も平均性能である場合には自然な目的関数である。

しかし、本論文が扱う問題では、テスト時に訓練分布と同じ混合比でデータが現れるとは限らない。特に、訓練データ中では多数派であるグループがテスト時にも支配的であるとは限らず、訓練データ上ではまれなグループがテスト時に重要になる可能性がある。このような分布変化を扱うために、分布ロバスト最適化（distributionally robust optimization; DRO）を考える。

DRO では、単一の訓練分布に対する平均損失ではなく、分布の不確実性集合 $\mathcal{Q}$ に含まれる分布のうち、最悪の場合の期待損失を最小化する。

$$
\underset{\theta\in\Theta}{\min}\Bigl\lbrace\mathcal{R}(\theta):=\underset{Q\in\mathcal{Q}}{\sup}\;\mathbb{E}_{(x,y)\sim Q}[\ell(\theta;(x,y))]\Bigr\rbrace.\tag{2}
$$

ここで、$\mathcal{Q}$ は潜在的なテスト分布の集合である。DRO は、この集合に含まれるどの分布に対しても損失が大きくならないようにモデルを学習する。したがって、DRO の性質は、不確実性集合 $\mathcal{Q}$ をどのように定義するかに強く依存する。

一般的には、訓練分布の周りのダイバージェンス球として $\mathcal{Q}$ を定義する方法も考えられる。しかし、そのような集合は広範な分布シフトを含む一方で、現実には起こりにくい極端な分布まで含んでしまう可能性がある。その場合、モデルは過度に悲観的な最悪ケース分布に対して最適化される。

そこで本論文では、疑似相関に関する事前知識を用いて訓練データをグループに分割し、そのグループに基づいて不確実性集合 $\mathcal{Q}$ を定義する。これが Group DRO の設定である。

訓練分布 $P$ は、$\mathcal{G}=\{1,2,\ldots,m\}$ で添字付けられた $m$ 個のグループ分布 $P_g$ の混合分布であるとする。このとき、不確実性集合 $\mathcal{Q}$ を、これらのグループ分布の任意の混合として定義する。

$$
\mathcal{Q}:=\left\{\sum_{g=1}^m q_g P_g : q\in\Delta_m\right\}.
$$

ここで、$\Delta_m$ は $(m-1)$ 次元の確率単体である。すなわち、

$$
\Delta_m=\left\{q\in\mathbb{R}^{m}:q_g\geq 0,\;\sum_{g=1}^{m}q_g=1\right\}
$$

である。

この定義では、潜在的なテスト分布は、グループ分布 $P_1,\ldots,P_m$ の混合比が変化したものとして表される。したがって、Group DRO は、入力空間全体に対する任意の分布シフトではなく、**グループ間の混合比が変わるような group shift を扱う** 方法である。

このとき、DRO の最悪ケースリスクは、各グループにおける期待損失の最大値として表される。

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))].\tag{3}
$$

つまり、Group DRO は平均リスクではなく、最も損失が大きいグループのリスクを小さくする。ERM が訓練分布上の平均損失を最小化するのに対して、**Group DRO はグループごとの損失の最大値を最小化する。**

実際には、真のグループ分布 $P_g$ は未知である。そのため、訓練データから得られる各グループの経験分布 $\hat{P}_g$ を用いる。各訓練点がどのグループに由来するかは既知であり、訓練データは $(x,y,g)$ の三つ組として与えられると仮定する。ただし、テスト時に $g$ を観測できるとは仮定しない。したがって、モデルはグループラベル $g$ を入力として直接利用するのではなく、訓練目的関数を構成するために利用する。

経験的な Group DRO の目的関数は次である。

$$
\hat{\theta}_{\mathrm{DRO}}:=\underset{\theta\in\Theta}{\argmin}\Bigl\lbrace \hat{\mathcal{R}}(\theta):=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]\Bigr\rbrace,\tag{4}
$$

ここで、$\hat{P}_g$ はグループ $g$ に属する訓練データ上の経験分布である。

式 (4) は、訓練データ上で最も損失が大きいグループを基準にモデルを学習することを意味する。したがって、Group DRO は、グループ間で良好な最悪グループ訓練損失を持つモデルを学習する方法である。

ただし、ここで注意すべき点がある。Group DRO が直接最小化しているのは、あくまで経験的な最悪グループリスク $\hat{\mathcal{R}}(\theta)$ である。一方、本当に評価したいのは、真の分布上の最悪グループリスク $\mathcal{R}(\theta)$ である。両者の間には、最悪グループ汎化ギャップが存在する。

$$
\delta:=\mathcal{R}(\theta)-\hat{\mathcal{R}}(\theta).
$$

したがって、Group DRO によって良好な最悪グループ訓練損失が得られたとしても、それだけで良好な最悪グループテスト損失が保証されるわけではない。この点が、本論文の中心的な問題意識である。特に、過剰パラメータ化されたニューラルネットワークでは、十分な正則化を適用しない限り、この汎化ギャップが大きくなり得る。

:::details なぜ Group DRO の最悪ケースリスクはグループリスクの最大値になるのか

Group DRO では、不確実性集合を

$$
\mathcal{Q}:=\left\{\sum_{g=1}^m q_g P_g : q\in\Delta_m\right\}
$$

と定義する。したがって、DRO の最悪ケースリスクは

$$
\sup_{Q\in\mathcal{Q}}\mathbb{E}_{(x,y)\sim Q}[\ell(\theta;(x,y))]
$$

である。この $Q$ は、グループ分布 $P_g$ の混合

$$
Q=\sum_{g=1}^{m}q_gP_g
$$

として表されるため、

$$
\begin{aligned}
\sup_{Q\in\mathcal{Q}}\mathbb{E}_{(x,y)\sim Q}[\ell(\theta;(x,y))]&=\sup_{q\in\Delta_m}\mathbb{E}_{(x,y)\sim \sum_{g=1}^{m}q_gP_g}[\ell(\theta;(x,y))] \\
&=\sup_{q\in\Delta_m}\sum_{g=1}^{m}q_g\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))].
\end{aligned}
$$

ここで、各グループのリスクを

$$
L_g(\theta):=\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

とおくと、

$$
\sup_{q\in\Delta_m}\sum_{g=1}^{m}q_gL_g(\theta)
$$

を評価すればよい。

これは $q$ に関する線形関数を確率単体 $\Delta_m$ 上で最大化する問題である。線形計画問題の最適解は、実行可能領域の頂点で達成される。確率単体 $\Delta_m$ の頂点は、ある1つのグループ $g$ にすべての質量を置く分布である。

したがって、

$$
\sup_{q\in\Delta_m}\sum_{g=1}^{m}q_gL_g(\theta)=\max_{g\in\mathcal{G}}L_g(\theta)
$$

である。すなわち、

$$
\sup_{Q\in\mathcal{Q}}\mathbb{E}_{(x,y)\sim Q}[\ell(\theta;(x,y))]=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

となる。これが式 (3) である。

> 小文字の $q$ はグループ混合比を表すベクトルであり、$Q$ はその混合比によって定まる 1 つの分布である。$\mathcal{Q}$ は、そのような分布 $Q$ をすべて集めた不確実性集合である。
:::

---

## 構成

| 章    | 本文の主張                                               | 主要数式                                | 画像                   | details                                             |
| ---- | --------------------------------------------------- | ----------------------------------- | -------------------- | --------------------------------------------------- |
| はじめに | 平均精度では atypical group の失敗が見えない                      | なし                                  | Figure 1             | なし                                                  |
| 1    | ERM / DRO / Group DRO の定式化                          | (1)(2)(3)(4)                        | なし                   | $\sup_{q\in\Delta_m}$ が $\max_g$ になる導出              |
| 2    | Group DRO は $g$ を入力に使わず、訓練目的に使う                     | $\mathcal{R}(\theta)$               | Figure 1 参照          | ERM と Group DRO のリスク比較                              |
| 3    | naive Group DRO は過剰パラメータ化領域で失敗する                    | $\hat{L}_{\mathrm{avg}}, \hat{L}_g$ | Table 1, Figure 2    | 平均訓練損失消失 $\Rightarrow$ worst-group training loss 消失 |
| 4    | 正則化は group-wise generalization gap を抑える             | $\delta_g(\theta)$                  | Table 1, Figure 2 参照 | $\mathcal{R}=\max_g(\hat{R}_g+\delta_g)$            |
| 5    | group adjustment は $\delta_g$ を $C/\sqrt{n_g}$ で見込む | (5)                                 | Table 2, Figure 3    | $C/\sqrt{n_g}$ の意味                                  |
| 6    | DRO は importance weighting と同じではない                  | (6)                                 | Table 3, Figure 4    | Proposition 1 と counterexample                      |
| 7    | Algorithm 1 で Group DRO を効率的に解く                     | (7)(8)(9)                           | なし                   | Proposition 2 の収束率                                  |
| 8    | 実験結果と結論を統合する                                        | 必要に応じて再掲なし                          | 既出図表を参照              | Appendix B の補足実験                                    |