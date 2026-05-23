---
title: "Group DRO 論文解説：過剰パラメータ化モデルの最悪グループ汎化"
emoji: "🐡"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["AI", "深層学習"]
published: true
---

## Group DRO 論文解説：過剰パラメータ化モデルの最悪グループ汎化

本記事では、Sagawa et al. による論文 *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization* を解説する。

この論文は、Group DRO を過剰パラメータ化されたニューラルネットワークに適用する際、単に worst-group training loss を最小化するだけでは不十分であり、worst-group generalization には正則化が重要であることを示した研究である。

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

## 2. Group DRO が見ているリスク

前章では、ERM、DRO、Group DRO の定式化を見た。ERM は訓練データ上の平均損失を最小化する。一方、Group DRO は、グループごとの損失の最大値を最小化する。

ここで重要なのは、Group DRO が「何を見ているのか」である。Group DRO は、グループラベル $g$ をモデルの入力として使う方法ではない。訓練時にグループラベル $g$ を用いて目的関数を構成し、どのグループで損失が大きいかを見ながらモデルを更新する方法である。

本論文では、各訓練点がどのグループに由来するかを知っていると仮定する。すなわち、訓練データは

$$
(x,y,g)
$$

の三つ組から構成される。ただし、テスト時に $g$ を観測できるとは仮定しない。したがって、モデルは $g$ を直接利用できない。Group DRO は、推論時に「このサンプルはどのグループか」を判定してから予測する方法ではなく、モデルが受け取る入力は通常どおり $x$ であり、予測対象は $y$ である。グループラベル $g$ は、学習時にどのグループの損失を重視するかを決めるために使われる。

したがって、Group DRO が見ているリスクは、入力 $x$ ごとの個別のリスクではなく、グループごとの集約リスクである。グループ $g$ に対するリスクは

$$
\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

であり、Group DRO はその最大値

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

を問題にする。

つまり、Group DRO は「平均的に良いモデル」ではなく、「最も悪いグループでも悪くなりすぎないモデル」を求める。この目的は、訓練データ中の多数派グループに引きずられる ERM とは異なる。

### グループはどのように定義されるか

本論文で扱う3つの応用では、各データ点 $(x,y)$ が、ラベル $y$ と疑似的に相関した入力属性

$$
a(x)\in\mathcal{A}
$$

を持つと考える。この $a(x)$ が疑似属性である。

グループは、疑似属性 $a$ とラベル $y$ の組によって定義される。すなわち、

$$
(a,y)
$$

の各値に対応してグループを作る。したがって、グループ数は

$$
m=|\mathcal{A}|\times|\mathcal{Y}|
$$

である。

この定義の意味は、単に属性 $a$ だけを見るのではなく、ラベル $y$ と属性 $a$ の組み合わせを見るという点にある。疑似相関の問題は、ラベルと属性の関係として現れるためである。

たとえば、Waterbirds では、ラベルは鳥の種類であり、疑似属性は背景である。waterbird は water background 上に、landbird は land background 上に現れやすいように訓練データが構成される。このとき、モデルは鳥そのものではなく背景に依存して分類する可能性がある。

CelebA では、ラベルは髪色であり、疑似属性は性別である。髪色と性別が訓練データ上で相関している場合、モデルは髪色そのものではなく、性別に由来する特徴を利用してしまう可能性がある。

MultiNLI では、ラベルは entailment, neutral, contradiction のいずれかであり、疑似属性は否定語の有無である。訓練データ中で否定語と contradiction が相関している場合、モデルは文の意味的関係ではなく、否定語の有無に依存する可能性がある。

この構造をまとめると、次のようになる。

| データセット | ラベル $y$ | 疑似属性 $a(x)$ | グループ |
|---|---|---|---|
| Waterbirds | waterbird / landbird | water background / land background | 鳥種 $\times$ 背景 |
| CelebA | blond / dark | male / female | 髪色 $\times$ 性別 |
| MultiNLI | entailment / neutral / contradiction | no negation / negation | NLI ラベル $\times$ 否定語の有無 |

Figure 1 で示されていたのは、このような構造である。訓練時には、ラベル $y$ と疑似属性 $a$ が相関している。しかし、テスト時にはその相関が成り立つとは限らない。このとき、疑似相関に依存したモデルは、相関が成り立つグループでは高い性能を示すが、相関が崩れるグループでは性能が低下する。

### Group DRO は何を防ごうとしているのか

ERM は、訓練分布上の平均損失を小さくする。したがって、訓練データ中で多数派を占めるグループの損失が目的関数に強く反映される。もし多数派グループで疑似相関が有効であれば、ERM はその疑似相関を利用するモデルを選びやすい。

一方、Group DRO は、グループごとの損失の最大値を小さくする。そのため、あるグループで疑似相関が成り立たず、損失が大きくなっている場合、そのグループが目的関数を支配する。モデルは、そのグループの損失を下げない限り、Group DRO 目的を改善できない。

この意味で、Group DRO は、疑似相関に依存した予測規則を避ける方向に働く。疑似相関は、多数派グループでは有効でも、少数派または非典型的なグループでは破綻する。そのため、最悪グループ損失を下げるには、疑似相関ではなく、よりグループ間で安定した特徴を使う必要がある。

ただし、これはあくまで目的関数上の期待である。Group DRO が最小化するのは、経験的な最悪グループリスク

$$
\hat{\mathcal{R}}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

である。したがって、Group DRO は最悪グループの訓練損失を直接制御するが、最悪グループのテスト損失を直接観測しているわけではない。

ここに、本論文の中心的な問題がある。Group DRO が見ているのは worst-group training risk である。一方、本当に欲しいのは **worst-group test risk** である。この2つが一致するためには、**各グループで汎化ギャップが十分に小さくなければならない。**

過剰パラメータ化されたニューラルネットワークでは、この点が問題になる。モデルが訓練データを全グループで完全に fit できる場合、Group DRO は訓練データ上ではうまくいっているように見える。しかし、それがテスト時の最悪グループ性能に結びつくとは限らない。次章では、この naive Group DRO の失敗がなぜ起こるのかを詳しく見る。

:::details ERM と Group DRO が最小化するリスクの違い

訓練分布 $P$ が、グループ分布 $P_g$ の混合として

$$
P=\sum_{g=1}^{m}p_gP_g
$$

と書けるとする。ここで、$p_g$ は訓練分布におけるグループ $g$ の混合比である。

このとき、ERM が対象とする期待損失は

$$
\mathbb{E}_{(x,y)\sim P}[\ell(\theta;(x,y))]
$$

である。$P$ をグループ分布の混合として展開すると、

$$
\begin{aligned}
\mathbb{E}_{(x,y)\sim P}[\ell(\theta;(x,y))]
&=\mathbb{E}_{(x,y)\sim \sum_{g=1}^{m}p_gP_g}[\ell(\theta;(x,y))] \\
&=\sum_{g=1}^{m}p_g\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))].
\end{aligned}
$$

したがって、ERM はグループごとのリスクを、訓練分布上の混合比 $p_g$ で重み付けした平均を最小化している。

一方、Group DRO は

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

を最小化する。

両者の違いは、グループリスクの集約方法にある。ERM は

$$
\sum_{g=1}^{m}p_g\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

という加重平均を見る。一方、Group DRO は

$$
\max_{g\in\mathcal{G}}\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

を見る。

そのため、ERM ではサンプル数の多いグループ、すなわち $p_g$ が大きいグループの損失が目的関数に強く反映される。一方、Group DRO では、混合比が小さいグループであっても、その損失が最大であれば目的関数を支配する。

この違いにより、ERM は平均性能を重視する目的関数であり、Group DRO は最悪グループ性能を重視する目的関数であると解釈できる。

:::

---

## 3. 過剰パラメータ化モデルで naive Group DRO が失敗する理由

前章までで、Group DRO は平均損失ではなく、グループごとの損失の最大値を小さくする目的関数であることを見た。したがって、一見すると、Group DRO を用いれば、疑似相関に依存したモデルを避け、最悪グループ性能を改善できるように思える。

しかし、本論文の重要な主張は、過剰パラメータ化されたニューラルネットワークに naive に Group DRO を適用しても、必ずしも最悪グループ性能は改善しないという点である。

理由は、過剰パラメータ化モデルが訓練データをほぼ完全に fit できるためである。モデルが訓練データ上の各サンプルをほぼ完全に分類できるなら、平均訓練損失は小さくなる。同時に、各グループにおける訓練損失も小さくなる。つまり、平均訓練損失が消失するモデルは、すでに最悪グループ訓練損失も消失させている。

この状況では、ERM と Group DRO の違いが訓練データ上では見えにくくなる。ERM は平均訓練損失を小さくする。一方、Group DRO は最悪グループ訓練損失を小さくする。しかし、モデルがすべての訓練点に fit できるなら、どちらの目的関数も同時に小さくなる。

したがって、過剰パラメータ化領域における問題は、worst-group training loss が大きいことではない。むしろ、訓練時には最悪グループでさえ高い性能を示しているにもかかわらず、テスト時には最悪グループ性能が低くなることが問題である。これは、グループごとの汎化ギャップが異なることを意味する。

### 3.1 ERM と DRO はどちらも訓練データには fit する

本論文では、Waterbirds と CelebA では ResNet50、MultiNLI では BERT を用いて、ERM と Group DRO の挙動を比較している。これらは、それぞれ画像分類および自然言語推論において高い平均テスト精度を達成する標準的なモデルである。

まず、標準的な正則化とハイパーパラメータ設定のもとで、モデルを収束するまで訓練する。この設定では、ERM モデルは3つのデータセットすべてにおいて、最悪グループ訓練精度で少なくとも 99.9% というほぼ完全な精度を達成している。さらに、平均テスト精度も Waterbirds、CelebA、MultiNLI でそれぞれ 97.3%、94.8%、82.5% と高い。

しかし、テスト時の最悪グループ精度は、それぞれ 60.0%、41.1%、65.7% まで低下する。平均テスト精度は高いにもかかわらず、最悪グループでは大きく失敗している。

Group DRO でも同様の現象が起こる。ERM モデルはほぼすべての訓練点を完全に分類しているため、ERM 目的だけでなく、DRO 目的に対してもほぼ最適になっている。実際、naive な Group DRO モデルも、ほぼ完全な訓練精度と高い平均テスト精度を達成する一方で、最悪グループテスト精度は低い。

![](https://static.zenn.studio/user-upload/b0fa671a711e-20260522.png)
*Table 1. 各訓練手法における平均精度と最悪グループ精度。正則化がない場合、ERM モデルと DRO モデルはいずれも最悪ケースグループで低い性能を示す。*

Table 1 の上段が、この現象を示している。standard regularization のもとでは、ERM と DRO はどちらも高い average accuracy を示す。しかし、worst-group accuracy は低い。これは、Group DRO が最悪グループ性能を改善するように設計されているにもかかわらず、過剰パラメータ化領域では naive に適用しても十分に機能しないことを示している。

ただし、これは Group DRO の目的関数そのものが無意味であることを意味しない。問題は、Group DRO が最小化している経験的な最悪グループリスクが、テスト時の最悪グループリスクと **一致していない** 点にある。

### 3.2 訓練性能の問題ではなく、汎化ギャップの問題である

Figure 2 は、CelebA における訓練中のグループごとの訓練精度と検証精度を示している。薄色の線が訓練精度、濃色の線が検証精度である。

![](https://static.zenn.studio/user-upload/c548c726ad2f-20260522.png)
*Figure 2. CelebA における訓練中の訓練精度と検証精度。デフォルト設定では ERM と DRO が全グループで完全な訓練精度を達成する一方、最悪ケースグループには悪く汎化する。*

Figure 2 の左側パネルを見ると、デフォルトのハイパーパラメータで収束まで訓練した場合、ERM モデルと DRO モデルはいずれも全グループでほぼ完全な訓練精度を達成している。しかし、検証精度を見ると、最悪ケースグループの性能は低い。

これは、最悪グループでの訓練性能が低いことが問題ではないことを示している。モデルは訓練時には、最悪グループでさえほぼ完全に fit している。それにもかかわらず、検証時にはそのグループで性能が低下する。

したがって、平均テスト精度と最悪グループテスト精度の差は、worst-group training performance の問題ではない。むしろ、グループごとの汎化ギャップが異なることから生じている。

ここで、真のグループリスクと経験グループリスクの差を考える。グループ $g$ に対する汎化ギャップを

$$
\delta_g(\theta)=\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]-\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

と書く。このとき、経験的なグループ損失が小さくても、$\delta_g(\theta)$ が大きければ、そのグループのテスト損失は大きくなる。

Group DRO が直接最小化しているのは、

$$
\hat{\mathcal{R}}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

である。一方、本当に評価したいのは、

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

である。

過剰パラメータ化モデルでは、$\hat{\mathcal{R}}(\theta)$ は小さくできる。しかし、それだけでは $\mathcal{R}(\theta)$ が小さいことは保証されない。特に、一部のグループで汎化ギャップが大きい場合、最悪グループテスト損失は大きくなる。

このため、naive Group DRO の失敗は、Group DRO が見ている訓練上の目的と、本当に改善したいテスト上の目的の間にギャップがあることから生じる。

### 3.3 なぜ ERM と naive DRO が同じように見えるのか

ERM と Group DRO は、本来異なる目的関数を持つ。ERM は平均訓練損失を最小化する。

$$
\hat{\theta}_{\mathrm{ERM}}=\underset{\theta\in\Theta}{\argmin}\;\mathbb{E}_{(x,y)\sim \hat{P}}[\ell(\theta;(x,y))]
$$

一方、Group DRO は最悪グループ訓練損失を最小化する。

$$
\hat{\theta}_{\mathrm{DRO}}=\underset{\theta\in\Theta}{\argmin}\left\{\underset{g\in\mathcal{G}}{\max}\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]\right\}
$$

通常であれば、平均損失を下げる解と、最悪グループ損失を下げる解は異なり得る。なぜなら、平均損失を下げるためには多数派グループの性能を優先すればよい一方で、最悪グループ損失を下げるには性能の悪いグループを改善しなければならないからである。

しかし、過剰パラメータ化モデルでは、訓練データを全グループでほぼ完全に fit できる。この場合、平均訓練損失も、最悪グループ訓練損失も同時に小さくなる。したがって、訓練データ上では ERM と Group DRO の目的関数の違いが **実質的に消えてしまう。**

この観点から見ると、naive Group DRO が失敗する理由は明確である。Group DRO は最悪グループ訓練損失を改善するための目的関数である。しかし、過剰パラメータ化領域では、最悪グループ訓練損失はすでに小さい。そのため、Group DRO は最悪グループテスト性能を改善するための追加的な制約として機能しにくい。

必要なのは、最悪グループ訓練損失を小さくすることだけではない。最悪グループ訓練損失がテスト性能に結びつくように、グループごとの汎化ギャップを制御することである。

この役割を担うのが、次章で扱う正則化である。

:::details 平均訓練損失が消えると worst-group training loss も消える

ここでは、過剰パラメータ化モデルにおいて、平均訓練損失が消失するモデルが、最悪グループ訓練損失も消失させることを確認する。

損失が非負であることを仮定する。すなわち、任意の訓練点 $(x_i,y_i)$ について、

$$
\ell(\theta;(x_i,y_i))\geq 0
$$

である。

全訓練データ上の平均経験損失を

$$
\hat{L}_{\mathrm{avg}}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(\theta;(x_i,y_i))
$$

とする。また、グループ $g$ の経験損失を

$$
\hat{L}_g(\theta)=\frac{1}{n_g}\sum_{i:g_i=g}\ell(\theta;(x_i,y_i))
$$

とする。ここで、$n_g$ はグループ $g$ に属する訓練点の数である。

損失の非負性より、

$$
\sum_{i:g_i=g}\ell(\theta;(x_i,y_i))\leq\sum_{i=1}^{n}\ell(\theta;(x_i,y_i))
$$

が成り立つ。したがって、

$$
\hat{L}_g(\theta)=\frac{1}{n_g}\sum_{i:g_i=g}\ell(\theta;(x_i,y_i))\leq\frac{1}{n_g}\sum_{i=1}^{n}\ell(\theta;(x_i,y_i))
$$

である。右辺は平均経験損失を用いて

$$
\frac{1}{n_g}\sum_{i=1}^{n}\ell(\theta;(x_i,y_i))=\frac{n}{n_g}\hat{L}_{\mathrm{avg}}(\theta)
$$

と書ける。よって、

$$
\hat{L}_g(\theta)\leq\frac{n}{n_g}\hat{L}_{\mathrm{avg}}(\theta)
$$

が得られる。

各グループのサンプル数 $n_g>0$ が固定されているとき、

$$
\hat{L}_{\mathrm{avg}}(\theta)\to 0
$$

ならば、

$$
\hat{L}_g(\theta)\to 0
$$

である。これは任意の $g\in\mathcal{G}$ について成り立つため、

$$
\max_{g\in\mathcal{G}}\hat{L}_g(\theta)\to 0
$$

となる。

したがって、平均訓練損失が消失する任意のモデルは、最悪グループ訓練損失も消失させる。

この事実は、過剰パラメータ化モデルにおいて naive Group DRO が ERM と差を出しにくい理由を説明している。ERM によって平均訓練損失がほぼ 0 になっているなら、そのモデルは経験的な DRO 目的に対してもほぼ最適になっているからである。

:::

---

## 4. 正則化の役割

前章では、過剰パラメータ化モデルに naive に Group DRO を適用しても、最悪グループ性能が改善しない場合があることを見た。理由は、モデルが訓練データを全グループでほぼ完全に fit できるためである。

この状況では、ERM と Group DRO の違いは訓練データ上では見えにくい。ERM は平均訓練損失を小さくする。Group DRO は最悪グループ訓練損失を小さくする。しかし、モデルがすべての訓練点をほぼ完全に分類できるなら、平均訓練損失も最悪グループ訓練損失も同時に小さくなる。

したがって、問題は worst-group training loss ではない。問題は、訓練データ上で小さい worst-group training loss が、テストデータ上の worst-group performance に結びついていないことである。言い換えると、問題の本体は group-wise generalization gap にある。

このギャップを制御するために、本論文が強調するのが正則化である。ここでいう正則化とは、モデルが訓練データを完全に fit することを防ぎ、汎化ギャップを小さくするための制約である。本論文では、主に強い $\ell_2$ ペナルティと早期終了が検討される。

### 4.1 強い $\ell_2$ ペナルティ

古典的には、正則化はモデル族の訓練データへの適合能力を制限し、汎化ギャップを制御するために用いられる。一方、現代の過剰パラメータ化領域では、明示的な正則化は平均性能にとって必須ではない場合がある。すべての正則化を取り除いても、モデルが平均的には良好に汎化することがあるためである。

しかし、本論文が示すのは、平均汎化と最悪グループ汎化では事情が異なるということである。平均性能にとっては強い正則化が不要に見える場合でも、最悪グループ性能にとっては正則化が重要になる。

ResNet50 における標準的な $\ell_2$ ノルムペナルティは

$$
\lambda\|\theta\|_2^2
$$

であり、デフォルトの係数は $\lambda=0.0001$ である。本論文では、$\lambda$ を数桁大きくし、Waterbirds では $\lambda=1.0$、CelebA では $\lambda=0.1$ とすることで、次の2つの効果が得られることを示している。

- 第一に、ERM モデルと DRO モデルのどちらも、完全な訓練精度を達成できなくなる。
- 第二に、各グループの汎化ギャップが大幅に小さくなる。

この変化が重要である。正則化が弱い場合、モデルは訓練データを全グループでほぼ完全に fit できる。そのため、ERM と DRO はどちらも訓練データ上では高い性能を示す。しかし、テスト時には最悪グループで大きく失敗する。

一方、強い $\ell_2$ ペナルティを用いると、モデルは訓練データを完全には fit できなくなる。このとき、ERM と DRO は異なる訓練上のトレードオフを行う。

ERM は平均訓練精度を高くするために、多数派グループを優先する。その結果、希少グループの訓練性能を犠牲にする。実際、強い $\ell_2$ ペナルティのもとで、ERM の最悪グループ訓練精度は Waterbirds で 35.7%、CelebA で 40.4% まで低下している。この場合、最悪グループテスト精度も低くなる。

一方、DRO は最悪グループ訓練精度を高く保つように学習する。強い $\ell_2$ ペナルティ領域では汎化ギャップが小さくなっているため、高い最悪グループ訓練精度が、高い最悪グループテスト精度に結びつく。その結果、Waterbirds では最悪グループテスト精度が 21.3% から 84.6% へ、CelebA では 37.8% から 86.7% へ改善する。

ここで重要なのは、DRO が単独で効いているわけではないという点である。DRO は最悪グループ訓練損失を下げる目的関数である。しかし、それがテスト時の最悪グループ性能に結びつくためには、グループごとの汎化ギャップが小さくなければならない。**強い $\ell_2$ ペナルティは、この汎化ギャップを制御する役割を持つ。**

Table 1 の中段は、この結果を示している。正則化が弱い場合、ERM と DRO のどちらも worst-group accuracy は低い。一方、強い $\ell_2$ ペナルティを入れると、DRO は ERM に対して最悪グループ精度を大きく改善する。

Figure 2 でも同じ構造が見える。強い $\ell_2$ ペナルティを用いた場合、ERM は希少グループの性能を犠牲にして平均性能を保つ。一方、DRO は全グループにわたって高い訓練精度と検証精度を達成する。

### 4.2 早期終了

本論文では、もう1つの正則化として早期終了も検討している。早期終了は、モデルが訓練データを完全に fit する前に学習を止めることで、過剰適合を抑える暗黙的な正則化である。

強い $\ell_2$ ペナルティの場合と同様に、早期終了も訓練損失が完全に消失する領域からモデルを遠ざける。これにより、汎化ギャップが小さくなり、Group DRO が最悪グループ訓練性能を高く保つことが、テスト時の最悪グループ性能に結びつきやすくなる。

本論文では、Section 3.1 と同じ設定を用いながら、各モデルを固定された少数のエポックだけ訓練している。この設定でも、DRO は ERM を最悪グループテスト精度で大きく上回る。

具体的には、早期終了を用いた場合、最悪グループテスト精度は Waterbirds で 6.7% から 86.0% へ、CelebA で 25.0% から 88.3% へ、MultiNLI で 66.0% から 77.7% へ改善する。平均テスト精度は ERM と DRO のどちらでも同程度に高いが、DRO では 1〜3% 程度の小さな低下が見られる。

ここでも、正則化の役割は明確である。早期終了は、モデルが訓練データを完全に fit することを防ぐ。すると、ERM と DRO は異なる訓練上の選択を行う。ERM は平均精度のために最悪グループを犠牲にしやすい。一方、DRO は最悪グループを改善するように学習する。

正則化なしでは、どちらの目的関数でも訓練損失が消えてしまうため、この違いは表面化しにくい。正則化によって完全な fit が妨げられることで、初めて ERM と DRO の目的関数の違いが実際の性能差として現れる。

### 4.3 正則化は Group DRO の補助ではなく、実質的な成立条件である

以上の結果から、本論文は、最悪グループ汎化における正則化の重要性を強調する。ここで注意すべきなのは、正則化が単なる性能改善のテクニックではないという点である。

Group DRO は、経験的な最悪グループリスク

$$
\hat{\mathcal{R}}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

を小さくする。しかし、評価したいのは、真の最悪グループリスク

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

である。

経験的な最悪グループリスクが小さくても、真の最悪グループリスクが小さいとは限らない。両者の差を決めるのが、グループごとの汎化ギャップである。

したがって、Group DRO が有効に働くためには、次の2つが同時に必要である。

- 1つ目は、Group DRO によって最悪グループ訓練損失を小さくすることである。
- 2つ目は、正則化によって、その訓練損失がテスト損失に結びつくように汎化ギャップを抑えることである。

正則化が不十分な場合、Group DRO は経験的な最悪グループリスクを小さくできる。しかし、汎化ギャップが大きければ、テスト時の最悪グループ性能は改善しない。したがって、過剰パラメータ化モデルにおいては、Group DRO と正則化は切り離して考えられない。

この点が、本論文の中心的なメッセージである。平均汎化を見る限り、過剰パラメータ化モデルに強い正則化は不要に見える場合がある。しかし、最悪グループ汎化を見ると、正則化は依然として重要である。つまり、平均性能と最悪グループ性能は、同じ汎化現象を見ているわけではない。

:::details worst-group test risk と group-wise generalization gap

Group DRO が経験的に最小化するのは、各グループの経験分布 $\hat{P}_g$ に基づく最悪グループリスクである。

$$
\hat{\mathcal{R}}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

一方、本当に評価したいのは、各グループの真の分布 $P_g$ に基づく最悪グループリスクである。

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

各グループの汎化ギャップを

$$
\delta_g(\theta)=\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]-\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

と定義する。このとき、各グループの真のリスクは

$$
\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]=\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\delta_g(\theta)
$$

と書ける。

したがって、最悪グループテストリスクは

$$
\mathcal{R}(\theta)=\underset{g\in\mathcal{G}}{\max}\left\{\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\delta_g(\theta)\right\}
$$

である。

この式から分かるように、Group DRO が経験的なグループ損失

$$
\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

を小さくしても、あるグループで $\delta_g(\theta)$ が大きければ、そのグループのテストリスクは大きくなる。

特に、過剰パラメータ化モデルでは、経験的なグループ損失を全グループで小さくできる。しかし、そのことは、グループごとの汎化ギャップが小さいことを意味しない。したがって、naive Group DRO は worst-group training risk を小さくできても、worst-group test risk を小さくできない場合がある。

正則化は、この $\delta_g(\theta)$ を抑える役割を持つ。強い $\ell_2$ ペナルティや早期終了によって、モデルが訓練データを完全に fit することを防ぐと、各グループの汎化ギャップが小さくなる。その結果、DRO が改善した worst-group training performance が、worst-group test performance に結びつきやすくなる。

:::

---

## 5. group adjustment：汎化ギャップを見込んだ補正

前章では、Group DRO と正則化を組み合わせることで、最悪グループ訓練性能を最悪グループテスト性能へ結びつける、という考え方を見た。Group DRO は最悪グループ訓練損失を小さくする。正則化は、グループごとの汎化ギャップを小さくする。両者を組み合わせることで、良好な worst-group training performance が、良好な worst-group test performance につながりやすくなる。

しかし、正則化を用いたとしても、汎化ギャップがすべてのグループで同じになるとは限らない。特に、グループサイズが大きく異なる場合、少数グループの汎化ギャップは多数グループより大きくなり得る。

本論文では、強い $\ell_2$ ペナルティを用いた Waterbirds の DRO モデルにおいて、最小グループの訓練精度とテスト精度のギャップが 15.4% である一方、最大グループでは 1.0% にとどまることが報告されている。これは、正則化によって全体的な汎化ギャップを抑えたとしても、グループ間で汎化ギャップの大きさが異なり得ることを示している。

したがって、単に経験的な最悪グループ損失を最小化するだけでは不十分な場合がある。訓練時点で、より大きな汎化ギャップを持つと予想されるグループについては、より低い訓練損失を要求する必要がある。これが group adjustment の考え方である。

### 5.1 グループごとの汎化ギャップを見込む

各グループ $g$ の汎化ギャップを

$$
\delta_g=\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]-\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

とする。ここで、$P_g$ はグループ $g$ の真の分布、$\hat{P}_g$ はグループ $g$ の経験分布である。

このとき、グループ $g$ の真のリスクは

$$
\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]=\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\delta_g
$$

と書ける。

したがって、最悪グループテスト損失は、経験損失だけでなく、グループごとの汎化ギャップも含めて

$$
\mathcal{R}(\theta)=\max_{g\in\mathcal{G}}\left\{\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\delta_g\right\}
$$

と見なせる。

しかし、実際には $\delta_g$ は未知である。そこで本論文では、汎化ギャップの単純な代理量として

$$
\hat{\delta}_g=\frac{C}{\sqrt{n_g}}
$$

を用いる。ここで、$n_g$ はグループ $g$ のサンプルサイズであり、$C$ はモデル容量に関する定数として扱われるハイパーパラメータである。

この補正は、小さいグループほど過学習しやすいという直感を反映している。一般に、モデル複雑度に基づく汎化境界では、サンプルサイズに対して $1/\sqrt{n}$ 型の項が現れる[^1]。そのため、グループサイズ $n_g$ が小さいほど、より大きな汎化ギャップを見込む。

この考え方に基づき、本論文では次の group-adjusted DRO estimator を導入する。

$$
\hat{\theta}_{\mathrm{adj}}:=\underset{\theta\in\Theta}{\argmin}\;\underset{g\in\mathcal{G}}{\max}\Bigl\lbrace\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\frac{C}{\sqrt{n_g}}\Bigr\rbrace.\tag{5}
$$

式 (5) は、経験的なグループ損失に対して、グループサイズに依存する補正項を加えている。グループサイズ $n_g$ が小さいほど、$C/\sqrt{n_g}$ は大きくなる。したがって、小さいグループは、経験損失が同じであっても、より大きな潜在的テスト損失を持つものとして扱われる。

[^1]: 古典的な学習理論における多くの汎化境界では、モデルクラスの複雑度や信頼水準を固定すると、経験リスクと真のリスクの差はサンプルサイズ $n$ に対して概ね $O(1/\sqrt{n})$ のスケールで減少する。<br>[Christoph Lampert, *Statistical Machine Learning*, Lecture Notes, IST Austria.](https://pub.ista.ac.at/~chl/courses/SML_W20/ml2020-08.pdf)

### 5.2 group adjustment は ERM ではなく Group DRO に効く

この補正は、Group DRO の設定だから意味を持つ。Group DRO では、グループごとの損失の最大値を取るため、

$$
\max_{g\in\mathcal{G}}\left\{\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]+\frac{C}{\sqrt{n_g}}\right\}
$$

という形で、各グループの補正量が最悪グループの選択に影響する。

一方、ERM は平均訓練損失を最小化する。もし ERM の目的関数に、グループごとの $C/\sqrt{n_g}$ を単純に加えても、これはモデルパラメータ $\theta$ に依存しない定数項になる。したがって、最適化には影響しない。

つまり、$C/\sqrt{n_g}$ による補正は、異なるサイズのグループに対する最悪グループ損失を考える Group DRO 設定でのみ機能する。

この点は重要である。group adjustment は単なる「少数グループの重みを上げる」操作ではない。最悪グループテスト損失の上界を意識して、経験的な最悪グループリスクを補正する操作である。

### 5.3 実験結果：group adjustment は worst-group accuracy を改善する

本論文では、強い $\ell_2$ ペナルティを用いた Group DRO モデルに対して、group adjustment の効果を評価している。対象は Waterbirds と CelebA である。

Waterbirds では、$\lambda=1.0$ の強い $\ell_2$ ペナルティを用いた Group DRO に group adjustment を加えることで、最悪グループテスト精度が 5.9% 改善し、誤差率は 3 分の 1 以上削減された。一方、CelebA では $\lambda=0.1$ において最悪グループ精度が 1.1% 向上した。CelebA での改善が小さいのは、$\ell_2$ ペナルティの効果がすでに大きく、グループ間の汎化ギャップのばらつきがそれほど大きくなかったためである。

![](https://static.zenn.studio/user-upload/7d49c61f5228-20260522.png)
*Table 2. グループ調整あり・なしの場合の平均テスト精度と最悪グループテスト精度。グループ調整は最悪グループ精度を改善するが、Waterbirds では平均精度が低下する。*

Table 2 は、group adjustment が最悪グループ精度を改善することを示している。ただし、Waterbirds では平均精度が低下している。これは、group adjustment が単純に全体性能を押し上げる操作ではないことを示している。

Group DRO は最悪グループ性能を重視する目的関数である。そこに group adjustment を加えると、小さいグループ、すなわち大きな汎化ギャップを持つと見込まれるグループをさらに重視する。その結果、最悪グループ精度は改善し得るが、平均精度とのトレードオフが生じる場合がある。

### 5.4 adjustment $C$ は汎化ギャップの見積もりを制御する

Figure 3 は、Waterbirds において、異なる調整値 $C$ を用いた場合のグループごとの訓練精度と検証精度の時間変化を示している。

![](https://static.zenn.studio/user-upload/19f07ba9b054-20260522.png)
*Figure 3. 異なる調整値 $C$ に対する、各グループの訓練精度と検証精度の時間変化。*

$C=0$ は、group adjustment を行わない場合である。このとき、land background 上の waterbirds の汎化ギャップが大きく、最悪グループ精度を押し下げている。

$C=2$ では、最悪グループ検証精度が最も良くなり、各グループの精度がバランスしている。これは、少数グループの汎化ギャップを適切に見込むことで、最悪グループ性能が改善することを示している。

一方、$C=4$ では補正が過剰になる。小さいグループ、たとえば land background 上の waterbirds は良好になるが、大きいグループ、たとえば land background 上の landbirds の性能が犠牲になる。つまり、$C$ が大きければよいわけではない。

この結果から、$C$ は単なる性能改善パラメータではなく、グループごとの汎化ギャップをどの程度見込むかを制御するハイパーパラメータであると解釈できる。小さすぎれば少数グループの汎化ギャップを補正できず、大きすぎれば少数グループを過剰に重視して他のグループの性能を犠牲にする。

### 5.5 group adjustment の位置づけ

group adjustment は、Group DRO の目的関数を、より test risk に近い形へ補正する操作である。

通常の Group DRO は、

$$
\max_{g\in\mathcal{G}}
\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

を最小化する。これは worst-group training loss を見ている。

一方、group adjustment は、

$$
\max_{g\in\mathcal{G}}
\left\{
\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
+
\frac{C}{\sqrt{n_g}}
\right\}
$$

を最小化する。これは、経験損失に対して、グループサイズに基づく汎化ギャップの見積もりを加えたものである。

したがって、group adjustment は、Group DRO の「最悪グループ訓練損失を小さくする」という目的を、「最悪グループテスト損失の推定上界を小さくする」という目的に近づけるための補正である。

この章までの流れをまとめると、Group DRO の問題は次のように整理できる。

まず、naive Group DRO は worst-group training loss を小さくする。しかし、過剰パラメータ化モデルでは、訓練損失が全グループで消えてしまうため、テスト時の最悪グループ性能には結びつかない。次に、正則化は group-wise generalization gap を抑える。さらに、group adjustment は、正則化後にも残るグループ間の汎化ギャップの違いを、グループサイズに基づいて補正する。

つまり、Group DRO、正則化、group adjustment は、それぞれ別の役割を持つ。

| 要素 | 役割 |
|---|---|
| Group DRO | worst-group training loss を小さくする |
| 正則化 | group-wise generalization gap を抑える |
| group adjustment | グループ間で異なる汎化ギャップを見込んで補正する |

このように見ると、group adjustment は Group DRO の周辺的な工夫ではなく、経験的な worst-group risk と真の worst-group risk のずれを補正するための自然な拡張である。

---

## 6. Group DRO と importance weighting の違い

前章までで、Group DRO は、最悪グループ訓練損失を小さくする目的関数であり、正則化および group adjustment と組み合わせることで、最悪グループ汎化を改善できることを見た。

ここで自然に生じる疑問は、Group DRO が単なる importance weighting、あるいは少数グループの upweighting と何が違うのか、という点である。実際、少数グループで性能が悪いなら、そのグループの重みを大きくして学習すればよいようにも見える。

原論文では、この疑問に対して、経験的比較と理論的比較の両方から答えている。結論を先に述べると、Group DRO は単なる固定重み付き ERM ではない。importance weighting はあらかじめ決めた重みで平均損失を最小化する。一方、Group DRO は、グループ分布の混合比を最悪ケース側が選ぶ min-max 問題である。

### 6.1 importance-weighted estimator

グループに対する重み $w\in\Delta_m$ を考える。ここで、$\Delta_m$ は $m$ 次元の確率単体である。importance weighting では、グループ $g$ に属するサンプルの損失に重み $w_g$ を掛けた平均損失を最小化する。

原論文では、importance-weighted estimator を次のように定義している。

$$
\hat{\theta}_w:=\underset{\theta\in\Theta}{\argmin}\;\mathbb{E}_{(x,y,g)\sim \hat{P}}[w_g\ell(\theta;x,y)].\tag{6}
$$

ここで、$\hat{P}$ は訓練データ上の経験分布であり、$w_g$ はグループ $g$ に対する重みである。

典型的な設計として、各グループの訓練頻度の逆数を重みに使う方法がある。すなわち、

$$
w_g=\frac{1}{\mathbb{E}_{g'\sim\hat{P}}[\mathbb{I}(g'=g)]}
$$

とする。この重み付けは、訓練データ中で少ないグループほど大きな重みを与えることに対応する。実装上は、各グループから等確率でサンプリングすることと近い。

この方法の直感は分かりやすい。少数グループが平均損失の中で埋もれるなら、少数グループの重みを大きくすればよい、という考え方である。

しかし、Group DRO が扱う問題は、単にグループ頻度が不均衡であるという問題ではない。グループごとにフィットしやすさが異なり、また各グループで汎化ギャップも異なり得る。そのため、少数グループを固定的に upweight しても、全グループで一様に低い損失が得られるとは限らない。

### 6.2 Group DRO は固定重みではなく adversarial な重みを見る

importance weighting と Group DRO の違いは、目的関数を見ると明確である。

importance weighting は、固定された重み $w_g$ のもとで

$$
\mathbb{E}_{(x,y,g)\sim \hat{P}}[w_g\ell(\theta;x,y)]
$$

を最小化する。これは、あらかじめ決めた重みによる平均損失最小化である。

一方、Group DRO は

$$
\underset{g\in\mathcal{G}}{\max}\;\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

を最小化する。あるいは、グループ上の分布 $q\in\Delta_m$ を用いると、

$$
\underset{q\in\Delta_m}{\sup}\sum_{g=1}^{m}q_g\mathbb{E}_{(x,y)\sim \hat{P}_g}[\ell(\theta;(x,y))]
$$

を内側に持つ min-max 問題として見られる。

つまり、importance weighting では重み $w$ は固定されている。一方、Group DRO では、損失が大きいグループに質量を置くように、内側の最悪ケース分布が選ばれる。この意味で、Group DRO のグループ重みは固定的ではなく、モデル $\theta$ に依存して **adversarial に決まる。**

この違いは重要である。ある固定重み $w$ が特定の状況では有効であっても、別のデータセットや別のモデルでは有効とは限らない。Group DRO は、現在のモデルに対してどのグループが最悪かを見ながら目的関数を構成するため、固定的な upweighting よりも直接的に worst-group risk を狙う。

### 6.3 経験的比較：upweighting は常に DRO の代替になるわけではない

原論文では、ERM、upweighting（UW）、Group DRO を比較している。各手法について、$\ell_2$ ペナルティ強度、エポック数、group adjustment にわたってグリッドサーチを行い、最も高い検証精度を示したモデルの結果を報告している。

![](https://static.zenn.studio/user-upload/6da9be0d16c3-20260522.png)
*Table 3. ERM、upweighting（UW）、group DRO モデルの比較。括弧内は二項標準偏差を表す。各目的関数について、$\ell_2$ ペナルティ強度、エポック数、グループ調整にわたってグリッドサーチを行い、最も高い検証精度を示したモデルの結果を報告している。*

Table 3 を見ると、Waterbirds と CelebA では、upweighting は ERM よりも worst-group accuracy を改善している。しかし、DRO はさらに高い worst-group accuracy を示している。

一方、MultiNLI では、upweighting は失敗している。ERM よりも低い average accuracy と worst-group accuracy しか達成できていない。これは、少数グループを upweight すれば常に最悪グループ性能が改善するわけではないことを示している。

原論文では、MultiNLI における upweighting の失敗について、希少グループが過度に強調され、そのグループに対して極端に低い訓練精度が得られる一方で、他のグループが犠牲になっているように見える、と述べている。これは、固定重みによる reweighting が、グループ間の難しさや最適化上のトレードオフを適切に扱えない可能性を示している。

この結果から、Group DRO は単なる minority upweighting ではないことが分かる。upweighting は、少数グループを大きく扱うヒューリスティックである。一方、Group DRO は、現在のモデルに対する最悪グループリスクを直接最小化する目的関数である。

### 6.4 凸設定では DRO と importance weighting は対応し得る

では、importance weighting によって、最適な worst-group risk を持つモデルを学習できるような重み $w$ は存在するのだろうか。

原論文では、凸設定では、ある分布 $Q^*\in\mathcal{Q}$ のもとで、DRO 解が重み付きリスク最小化の解として表せることを示している。これは、DRO と importance weighting が、少なくとも凸設定では対応し得ることを意味する。

直感的には、DRO の最適解 $\theta^*$ に対して、最悪ケース分布 $Q^*$ が存在し、その $Q^*$ のもとでの期待損失を最小化しても $\theta^*$ が得られる、ということである。

ただし、ここで重要なのは、これは「任意の単純な重みでよい」という主張ではない点である。対応する重みは、最適な DRO 解 $\theta^*$ や最悪ケース分布 $Q^*$ と関係している。したがって、実際にその重みを事前に知ることは容易ではない。

さらに、深層ニューラルネットワークのような非凸モデルでは、この対応は一般には成り立たない。

::::details 凸設定における DRO と importance weighting の関係

ここでは、原論文の Appendix A.1 の命題 1 に対応する議論を整理する。

すべての $z\in\mathcal{Z}$ に対して、損失 $\ell(\cdot;z)$ が連続かつ凸であるとする。また、不確実性集合 $\mathcal{Q}$ とモデル族 $\Theta\subseteq\mathbb{R}^d$ が凸かつコンパクトであるとする。

関数

$$
h(\theta,Q):=\mathbb{E}_{z\sim Q}[\ell(\theta;z)]
$$

を定義する。このとき、$h(\theta,Q)$ は $\theta$ に関して凸であり、$Q$ に関して線形、したがって凹である。

DRO 目的は

$$
\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

である。仮定のもとで、max-min 等式により

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)=\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

が成り立つ。

:::details なぜ min と max の順番を入れ替えられるのか

ここで用いている等式

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)=\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

は、常に成り立つわけではない。一般には、

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)\leq\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

のみが成り立つ。

この不等式は、$\theta$ と $Q$ のどちらを先に選ぶかの違いとして理解できる。

右辺

$$
\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

では、まずモデル側が $\theta$ を選び、その後で adversary 側がその $\theta$ にとって最も損失が大きくなる分布 $Q$ を選ぶ。これは DRO の目的そのものであり、モデル側にとって厳しい順序である。

一方、左辺

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)
$$

では、まず分布 $Q$ が固定され、その後でモデル側がその $Q$ に対して最適な $\theta$ を選べる。つまり、モデルは分布を見た後で最適化できる。したがって、こちらの方がモデル側にとって有利である。

そのため、一般には

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)\leq\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

となる。

しかし、Appendix A.1 では、この不等式が等号になるための条件を置いている。具体的には、$\Theta$ と $\mathcal{Q}$ が凸かつコンパクトであり、関数 $h(\theta,Q)$ が $\theta$ に関して凸、$Q$ に関して凹であると仮定する。このような凸・凹構造のもとでは、minimax theorem により、min と max の順序を入れ替えることができる[^2]。

今回の設定では、

$$
h(\theta,Q)=\mathbb{E}_{z\sim Q}[\ell(\theta;z)]
$$

である。

まず、$\theta$ に関する凸性を確認する。各 $z$ に対して損失 $\ell(\cdot;z)$ が $\theta$ に関して凸であるので、その期待値である $h(\theta,Q)$ も $\theta$ に関しても凸である。これは、凸関数の非負重み付き平均も凸関数になるためである。

次に、$Q$ に関する性質を確認する。任意の $Q_1,Q_2\in\mathcal{Q}$ と $\lambda\in[0,1]$ に対して、混合分布

$$
\lambda Q_1+(1-\lambda)Q_2
$$

を考える。この分布は、確率 $\lambda$ で $Q_1$ からサンプルし、確率 $1-\lambda$ で $Q_2$ からサンプルする分布である。このとき、期待値の線形性より、

$$
\begin{aligned}
h(\theta,\lambda Q_1+(1-\lambda)Q_2)
&=\mathbb{E}_{z\sim \lambda Q_1+(1-\lambda)Q_2}[\ell(\theta;z)] \\
&=\lambda \mathbb{E}_{z\sim Q_1}[\ell(\theta;z)]+(1-\lambda)\mathbb{E}_{z\sim Q_2}[\ell(\theta;z)] \\
&=\lambda h(\theta,Q_1)+(1-\lambda)h(\theta,Q_2).
\end{aligned}
$$

したがって、$\theta$ を固定したとき、$h(\theta,Q)$ は $Q$ に関して線形である。

> 分布を混合すると、期待損失も同じ比率で混合されるという意味

線形関数は、凸性の不等式と凹性の不等式をどちらも等号で満たす。したがって、線形関数は凸でもあり凹でもある。今回の minimax theorem では、最大化側の変数 $Q$ に関して凹性が必要であるため、$Q$ に関する線形性はその条件を満たしている。

以上より、今回の仮定のもとでは、$h(\theta,Q)$ は $\theta$ に関して凸、$Q$ に関して凹である。したがって minimax theorem を適用でき、

$$
\sup_{Q\in\mathcal{Q}}\inf_{\theta\in\Theta}h(\theta,Q)=\inf_{\theta\in\Theta}\sup_{Q\in\mathcal{Q}}h(\theta,Q)
$$

が成り立つ。

[^2]: [Maurice Sion, “On General Minimax Theorems,” *Pacific Journal of Mathematics*, 8(1), 171–176, 1958.](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-8/issue-1/On-general-minimax-theorems/pjm/1103040253.full)

:::

したがって、ある鞍点 $(\theta^*,Q^*)$ が存在し、

$$
\sup_{Q\in\mathcal{Q}}h(\theta^*,Q)=h(\theta^*,Q^*)=\inf_{\theta\in\Theta}h(\theta,Q^*)
$$

を満たす。

最後の等式

$$
h(\theta^*,Q^*)=\inf_{\theta\in\Theta}h(\theta,Q^*)
$$

は、DRO 解 $\theta^*$ が、ある分布 $Q^*$ のもとでの期待損失

$$
\mathbb{E}_{z\sim Q^*}[\ell(\theta;z)]
$$

も最小化していることを意味する。

したがって、凸設定では、DRO 解をある importance weighting の解として表せる場合がある。

ただし、この結果は、DRO と任意の固定的な upweighting が同じであることを意味しない。対応する $Q^*$ は DRO 問題の鞍点として得られるものであり、単純な逆頻度重みなどと一致するとは限らない。

::::

### 6.5 非凸設定では等価性が崩れる

深層ニューラルネットワークは非凸モデルである。そのため、凸設定で成立する DRO と importance weighting の対応は、そのまま適用できない。

原論文では、Figure 4 の toy example によって、非凸設定では DRO 解がどの importance weighting の最小化解としても得られない場合があることを示している。

![](https://static.zenn.studio/user-upload/5e5ca2f3105c-20260522.png)
*Figure 4. DRO と importance weighting が等価ではないことを示す toy example。DRO 解は $\theta^*$ である一方、任意の importance weighting は $\theta_1$ または $\theta_2$ における解を与える。*

この例では、2点

$$
\mathcal{Z}=\{z_1,z_2\}
$$

上に支持を持つ一様なデータ分布 $P$ を考える。また、パラメータ空間を

$$
\Theta=[0,1]
$$

とする。損失 $\ell(\theta;z)$ は Figure 4 のような非凸形状を持つ。

DRO は、2つの点に対する損失の最悪ケースを小さくするため、解 $\theta^*$ を選ぶ。この解では、最悪ケース損失が

$$
\mathcal{R}(\theta^*)=0.6
$$

になる。

一方、任意の重み

$$
(w_1,w_2)\in\Delta_2
$$

を考える。一般性を失わず、

$$
w_1\geq w_2
$$

と仮定すると、重み付き損失

$$
w_1\ell(\theta;z_1)+w_2\ell(\theta;z_2)
$$

の最小化解は $\theta_1$ になる。このとき、最悪ケース損失は

$$
\mathcal{R}(\theta_1)=1.0
$$

であり、DRO 解 $\theta^*$ より悪い。$w_2\geq w_1$ の場合も対称的に、解は $\theta_2$ になる。

したがって、この toy example では、任意の importance weighting によっても DRO 解 $\theta^*$ は得られない。これは、非凸設定では、DRO が固定重み付きリスク最小化に還元できない場合があることを示している。

:::details Figure 4 の counterexample の意味

Figure 4 の要点は、DRO と importance weighting の違いが、単なる実装上の違いではなく、最適化問題としての違いであることを示す点にある。

DRO は

$$
\min_{\theta\in\Theta}\max_{Q\in\mathcal{Q}}\mathbb{E}_{z\sim Q}[\ell(\theta;z)]
$$

を解く。2点 $\mathcal{Z}=\{z_1,z_2\}$ の場合、これは直感的には

$$
\min_{\theta\in\Theta}\max\{\ell(\theta;z_1),\ell(\theta;z_2)\}
$$

に対応する。

したがって、DRO は2つの損失のうち大きい方をできるだけ小さくする点を選ぶ。Figure 4 では、この解が $\theta^*$ である。

一方、importance weighting は、固定された重み $(w_1,w_2)$ に対して

$$
\min_{\theta\in\Theta}\left\{w_1\ell(\theta;z_1)+w_2\ell(\theta;z_2)\right\}
$$

を解く。これは、2つの損失の加重平均を最小化する問題である。

凸設定であれば、min-max 問題と適切な重み付き平均問題が対応する場合がある。しかし、非凸設定では、重み付き平均の大域的最小解が、min-max 解と一致するとは限らない。

Figure 4 では、任意の重みに対して、重み付き平均の最小解は $\theta_1$ または $\theta_2$ になる。一方、DRO 解は $\theta^*$ である。したがって、どの固定重みを使っても、DRO 解を得ることができない。

この反例が示しているのは、DRO が単なる reweighting の別表現ではないということである。少なくとも非凸設定では、DRO は固定重み付き ERM とは異なる解を選び得る。

:::

### 6.6 まとめ：Group DRO は少数派 upweighting ではない

この章の結論は明確である。Group DRO は、少数派グループを単に大きく重み付けする方法ではない。

importance weighting は、固定された重み $w_g$ のもとで平均損失を最小化する。逆頻度重みのような方法では、少数グループが大きく扱われる。しかし、これはあくまで固定的なヒューリスティックである。グループ間の難しさや、モデルがどのグループで現在失敗しているかを直接反映するわけではない。

一方、Group DRO は、グループ分布の混合に対する最悪ケースを考える。つまり、現在のモデルにとって最も損失が大きいグループを **adversarial** に重視する。したがって、Group DRO は worst-group risk を直接扱う min-max 最適化である。

経験的にも、Table 3 では upweighting が Waterbirds と CelebA では ERM より改善する一方、MultiNLI では失敗している。理論的にも、Figure 4 のように、非凸設定では DRO 解が任意の importance weighting の解として得られない場合がある。

したがって、Group DRO は「少数派に重みをかけるだけ」の方法ではない。Group DRO は、グループシフトに対する worst-case risk を明示的に定式化し、その最悪ケースを下げるための目的関数である。

---

## 7. Group DRO の最適化アルゴリズム

ここまで、Group DRO の目的関数、過剰パラメータ化モデルにおける naive Group DRO の限界、正則化、group adjustment、importance weighting との違いを見てきた。最後に、Group DRO をどのように最適化するかを整理する。

Group DRO の目的は、グループごとの損失の最大値を小さくすることである。これは、グループ上の分布 $q\in\Delta_m$ を用いると、min-max 問題として書ける。原論文では、最適化問題 (4) を次のように書き換えている。

$$
\underset{\theta\in\Theta}{\min}\;\underset{q\in\Delta_m}{\sup}\sum_{g=1}^m q_g \mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))].\tag{7}
$$

ここで、$\theta$ はモデルパラメータ、$q$ はグループ上の分布である。$q_g$ はグループ $g$ に割り当てられる質量を表す。内側の $\sup$ は、現在のモデル $\theta$ に対して損失が大きいグループに重みを置く操作である。

この式の意味は単純である。モデル側は損失を小さくしようとする。一方、分布側は損失が大きいグループに質量を置き、モデルにとって悪いグループを強調する。したがって、Group DRO は、モデルパラメータ $\theta$ とグループ重み $q$ の二者による min-max 最適化として理解できる。

### 7.1 既存の DRO 最適化との違い

DRO に対する既存の確率的最適化手法には、大きく2つの方向がある。1つは、目的関数のラグランジュ双対に対して SGD を行う方法である。もう1つは、minimax 問題を直接最適化する方法である。

しかし、原論文では、前者は group DRO ではうまく機能しにくいと説明されている。理由は、双対目的関数の勾配を確率的かつ不偏に推定することが難しいためである。後者の minimax 型アルゴリズムについても、既存手法では収束保証がなく、一部の設定で不安定性が観察されたとされている。

そこで原論文では、$\theta$ と $q$ に対する勾配ベースの更新を交互に行うオンライン最適化アルゴリズムを導入する。直感的には、グループ上の分布 $q$ を保持し、損失の大きいグループに大きな質量を割り当てる。そして、各例について、その例が属するグループの質量に比例して $\theta$ を更新する。

具体的には、$\theta$ に対しては SGD を行い、$q$ に対しては exponentiated gradient ascent を行う。

### 7.2 Algorithm 1：Group DRO のオンライン最適化

原論文の Algorithm 1 は、次のような更新を行う。

:::message 
**Algorithm 1:** Online optimization algorithm for group DRO
**Input:** Step sizes $\eta_q, \eta_\theta; P_g$ for each $g\in\mathcal{G}$
Initialize $\theta^{(0)}$ and $q^{(0)}$
**for** $t=1,\ldots,T$ **do**
$|\quad g\sim\mathrm{Uniform}(1,\ldots,m)\quad$ 
$|\quad$ // グループ $g$ を一様に選ぶ
$|\quad$
$|\quad x,y\sim P_g\quad$ 
$|\quad$ // グループ $g$ からサンプルを得る
$|\quad$
$|\quad q'\leftarrow q^{(t-1)};q'_g\leftarrow q'_g\exp(\eta_q\ell(\theta^{(t-1)};(x,y)))\quad$ 
$|\quad$ // 損失が大きいほど、グループ $g$ の重みを増やす
$|\quad$
$|\quad q^{(t)}\leftarrow q'/\sum_{g'} q'_{g'}\quad$
$|\quad$ // $q$ を確率分布に正規化する
$|\quad$
$|\quad \theta^{(t)}\leftarrow\theta^{(t-1)}-\eta_\theta q_g^{(t)}\nabla\ell(\theta^{(t-1)};(x,y))\quad$
$|\quad$ // グループ重み $q_g$ に比例して $\theta$ を更新する
**end**
:::

このアルゴリズムでは、まずグループ $g$ を一様にサンプルし、そのグループからデータ点 $(x,y)$ をサンプルする。そして、そのデータ点に対する損失

$$
\ell(\theta^{(t-1)};(x,y))
$$

に応じて、グループ重み $q_g$ を指数的に増やす。具体的には、

$$
q'_g\leftarrow q'_g\exp(\eta_q\ell(\theta^{(t-1)};(x,y)))
$$

という更新を行う。

この更新は、損失が大きいグループほど次の反復で大きな重みを持つようにする。更新後の $q'$ は、そのままでは確率分布でなくなるため、

$$
q^{(t)}\leftarrow\frac{q'}{\sum_{g'}q'_{g'}}
$$

として正規化する。

その後、モデルパラメータ $\theta$ を更新する。更新式は

$$
\theta^{(t)}\leftarrow\theta^{(t-1)}-\eta_\theta q_g^{(t)}\nabla\ell(\theta^{(t-1)};(x,y))
$$

である。つまり、選ばれたサンプルの損失勾配に対して、そのサンプルが属するグループの重み $q_g^{(t)}$ を掛けて更新する。

この仕組みにより、現在損失が大きいグループのサンプルは、より大きな影響をモデル更新に与える。一方で、損失が小さいグループは相対的に小さな重みを持つ。

### 7.3 hard な worst group selection ではなく、$q$ を勾配で更新する

このアルゴリズムの重要な点は、各反復で平均損失が最大のグループを hard に選ぶのではなく、グループ分布 $q$ を勾配ベースで更新することである。

もし毎回、現在の損失が最大のグループだけを選んで更新すると、推定ノイズやミニバッチの揺らぎによって、更新対象が不安定になりやすい。特に、深層ニューラルネットワークの訓練では、損失推定には大きなばらつきがある。そのため、hard に worst group を選ぶ方法は不安定になり得る。

これに対して、exponentiated gradient ascent による $q$ の更新では、損失が大きいグループの重みを滑らかに増やす。$q$ は確率単体 $\Delta_m$ 上に保たれるため、全グループに対する重み分布として解釈できる。

したがって、この方法は、最悪グループを明示的に1つ選ぶというよりも、損失の大きいグループに徐々に質量を移していく方法である。この滑らかな重み更新が、安定性と収束保証を得るうえで重要になる。

原論文でも、既存の group DRO アルゴリズムからの主要な改善点は、各イテレーションで平均損失が最悪のグループを選ぶのではなく、勾配を用いて $q$ を更新する点であると説明されている。

### 7.4 Algorithm 1 の意味と計算コスト

Algorithm 1 は、Group DRO の min-max 目的を、大規模モデルでも実装可能な形に落とし込むための手続きである。

Group DRO の本質は、損失が大きいグループを重視することである。しかし、各ステップで全グループの正確なリスクを計算し、その中から最悪グループを選ぶことは、大規模データセットでは現実的でない。また、ミニバッチ推定に基づいて hard に最悪グループを選ぶと、損失推定の揺らぎに過敏に反応し、最適化が不安定になりやすい。

そこで Algorithm 1 では、最悪グループを毎回 hard に選ぶのではなく、グループ上の分布 $q$ をオンラインに更新する。損失が大きいグループには指数的に重みを増やし、その重み $q_g$ に比例してモデルパラメータ $\theta$ を更新する。これにより、現在のモデルにとって損失が大きいグループを継続的に重視しながら、更新自体は通常の SGD に近い形で実行できる。

この点で、Algorithm 1 は単なる重み付き ERM ではない。固定された重みで平均損失を最小化するのではなく、学習の進行に応じてグループ重み $q$ を更新し、モデルにとって悪いグループへ質量を移していく。つまり、Group DRO の **グループ上の adversarial distribution を持つ min-max 最適化** を、オンラインかつスケーラブルに近似するアルゴリズムである。

計算コストの観点でも、この設計は重要である。Group DRO は min-max 問題であるため、一見すると通常の ERM より大きな計算コストを必要とするように見える。しかし Algorithm 1 では、モデル更新そのものは通常の SGD と同じく、サンプルごとの損失と勾配に基づいて行われる。追加で必要なのは、グループ重み $q$ の exponentiated gradient 更新と、確率単体 $\Delta_m$ 上に戻すための正規化である。

原論文では、実行時間の大部分は損失とその勾配の計算に支配されるため、エポック数を固定した場合、通常の SGD との差は 5% 未満であると説明されている。したがって、グループ数 $m$ が極端に大きくなければ、Group DRO は通常の深層学習パイプラインに近い計算コストで実装できる。

まとめると、Algorithm 1 の役割は、Group DRO の理論的な min-max 目的を、通常の SGD に近い計算量で実行可能な訓練手続きへ変換することにある。hard な worst-group selection を避け、$q$ を滑らかに更新することで安定性を確保しつつ、損失の大きいグループを動的に重視する。これが、Group DRO を大規模なニューラルネットワークに適用可能にしている。

:::details 原論文 Proposition 2: 凸設定における収束保証

原論文の命題 2 は、Algorithm 1 の平均反復点が、凸設定において minimax 最適値へ近づくことを示している。

まず、

$$
L(\theta,q):=\sum_{g=1}^m q_g\mathbb{E}_{(x,y)\sim P_g}[\ell(\theta;(x,y))]
$$

と定義する。これは、グループ分布 $q$ のもとでの期待損失である。

Group DRO の期待目的は、

$$
\min_{\theta\in\Theta}\max_{q\in\Delta_m}L(\theta,q)
$$

である。

Algorithm 1 の平均反復点を $\bar{\theta}^{(1:T)}$ とする。このとき、誤差 $\epsilon_T$ は

$$
\epsilon_T=\underset{q\in\Delta_m}{\max}\;L(\bar{\theta}^{(1:T)},q)-\underset{\theta\in\Theta}{\min}\;\underset{q\in\Delta_m}{\max}\;L(\theta,q)
$$

と定義される。

第一項は、アルゴリズムが出力した平均反復点に対する worst-case loss である。第二項は、理想的な Group DRO 最適値である。したがって、$\epsilon_T$ が小さいほど、Algorithm 1 の出力は minimax 最適解に近い。

命題 2 は、次の仮定のもとでこの誤差を評価する。

* 損失 $\ell(\cdot;(x,y))$ は非負である。
* 損失 $\ell(\cdot;(x,y))$ は凸である。
* 損失 $\ell(\cdot;(x,y))$ は $B_\nabla\text{-Lipschitz}$ 連続である。
* 損失は $B_\ell$ で有界である。
* パラメータ集合 $\Theta$ は凸であり、任意の $\theta\in\Theta$ について $\|\theta\|_2\leq B_\Theta$ が成り立つ。

このとき、

$$
\mathbb{E}[\epsilon_T]\leq 2m\sqrt{\frac{10(B_\Theta^2 B_\nabla^2 + B_\ell^2 \log m)}{T}}\tag{9}
$$

である。

この式から、期待誤差は反復回数 $T$ に対して

$$
O\left(\frac{1}{\sqrt{T}}\right)
$$

で減少することが分かる。

また、境界にはグループ数 $m$ が係数として現れる。これは、Group DRO がグループ上の分布 $q\in\Delta_m$ を最適化する問題であり、グループ数が多いほど adversarial distribution の探索が難しくなることを反映している。

ただし、この命題は凸設定に対する保証である。深層ニューラルネットワークでは損失は一般に非凸であるため、この結果は直接的な大域収束保証ではない。それでも、Algorithm 1 がどのような意味で Group DRO の minimax 目的を近似しているのかを理解するための理論的基盤になる。

:::

---

## 8. 実験結果と論文の結論

ここまで、Group DRO の定式化、過剰パラメータ化モデルにおける naive Group DRO の失敗、正則化、group adjustment、importance weighting との差、そして最適化アルゴリズムを見てきた。

本章では、原論文の実験結果と議論を総括する。重要なのは、個々の表や図を独立した結果として読むのではなく、論文全体の主張として接続して理解することである。

本論文の結論は、次の一文に集約できる。

過剰パラメータ化モデルにおいて、Group DRO は単独では最悪グループ汎化を十分に改善しない。最悪グループ性能を改善するには、Group DRO によって worst-group training loss を重視するだけでなく、正則化によって group-wise generalization gap を制御する必要がある。

### 8.1 ERM と naive Group DRO：平均精度では見えない失敗

Figure 1 で示されていたように、本論文が扱う3つのタスクでは、訓練時にラベル $y$ と疑似属性 $a$ が相関している。しかし、テスト時にはその相関が成り立つとは限らない。この状況では、モデルが疑似相関に依存していても、多数派グループでは高い性能を示すため、平均精度だけを見ると問題が見えにくい。

Table 1 の上段は、この問題を数値として示している。ERM は高い average accuracy を達成する一方で、worst-group accuracy は大きく低下している。これは、平均性能が高いことが、すべてのグループで性能が高いことを意味しないことを示している。

Group DRO は、この問題に対して worst-group training loss を最小化する自然な目的関数を与える。しかし、naive Group DRO は過剰パラメータ化領域では十分に機能しない。Table 1 と Figure 2 が示すように、ERM も Group DRO も訓練データにはほぼ完全に fit している。それにもかかわらず、最悪グループのテスト性能は低い。

したがって、問題は worst-group training performance ではない。訓練時には最悪グループでさえ高い性能を示しているためである。問題の本体は、**グループごとの汎化ギャップ** である。naive Group DRO が効かないのは、worst-group objective が不要だからではなく、経験的な worst-group objective が小さくなっても、それが test performance に結びつかないからである。

ここで正則化が必要になる。Table 1 の中段・下段と Figure 2 は、強い $\ell_2$ ペナルティや早期終了を用いると、Group DRO が ERM に対して worst-group accuracy を大きく改善することを示している。正則化によって訓練データへの完全な fit が抑えられると、ERM と Group DRO は異なるトレードオフを行う。ERM は平均性能を優先し、Group DRO は最悪グループ性能を優先する。この違いが、正則化下ではじめてテスト性能の差として現れる。

したがって、正則化は Group DRO の単なる補助ではない。過剰パラメータ化モデルにおいて、Group DRO を worst-group generalization に結びつけるための **中心的な条件** である。

### 8.2 補正・比較・最適化：Group DRO を実際に機能させる要素

正則化によって汎化ギャップを抑えても、グループ間で汎化ギャップの大きさが完全に揃うわけではない。そこで原論文では、**group adjustment** を導入している。これは、小さいグループほど大きな汎化ギャップを持つと見込み、経験的なグループ損失に補正を加える方法である。

Table 2 は、group adjustment によって worst-group accuracy が改善することを示している。Figure 3 は、調整値 $C$ の効果を示している。$C$ が小さすぎれば少数グループの汎化ギャップを補正できず、大きすぎれば少数グループを過剰に重視して他のグループを犠牲にする。したがって、group adjustment は、少数グループを機械的に優先する操作ではなく、グループサイズに応じて経験損失とテスト損失のずれを見込む補正である。

また、Group DRO は importance weighting とも異なる。Table 3 は、upweighting が Waterbirds と CelebA では ERM より改善する一方、MultiNLI では失敗することを示している。一方、Group DRO は三つのデータセット全体でより安定した worst-group accuracy を示している。

この結果は、Group DRO が単なる minority upweighting ではないことを示している。固定重みによって少数グループを大きく扱っても、グループ間の難しさや汎化ギャップの違いを適切に処理できるとは限らない。理論的にも、Figure 4 は、非凸設定では DRO 解が任意の importance weighting の最小化解として得られない場合があることを示している。

最後に、Algorithm 1 は Group DRO を大規模ニューラルネットワークに適用するための最適化手続きである。モデルパラメータ $\theta$ は SGD で更新し、グループ上の分布 $q$ は exponentiated gradient ascent で更新する。重要なのは、各ステップで hard に最悪グループを選ぶのではなく、損失の大きいグループへ滑らかに質量を移していく点である。これにより、ミニバッチ損失の揺らぎに過敏に反応せず、通常の SGD に近い計算コストで Group DRO の min-max 目的を扱える。

:::details グループ指定が不完全な場合の補足実験

原論文の補足実験では、CelebA において、強い $\ell_2$ ペナルティを用いた Group DRO モデルを再評価している。ただし、グループ指定を意図的に不完全にしている。

具体的には、ground-truth の疑似属性である Male の代わりに、関連する属性 Wearing Lipstick を用いる。さらに、Eyeglasses、Smiling、Double Chin、Oval Face という4つの distractor attribute を追加する。

このとき、ラベルと複数属性の組み合わせにより、多数のグループを構成する。原論文では、この設定でも Group DRO が元の4グループに対して高い robust accuracy を維持できることが示されている。

この結果は、Group DRO が完全な spurious attribute の指定にのみ依存するわけではないことを示唆している。ただし、性能は正確な group specification を用いた場合よりも低下する。したがって、**Group DRO はグループ指定の不完全性にある程度は頑健であるが、どの属性でグループを定義するかは依然として重要である。**

:::

### 8.3 論文の結論：worst-group generalization における正則化の重要性

以上を踏まえると、本論文の結論は次のように整理できる。

1. 平均精度はモデルの信頼性を十分に表さない。モデルは i.i.d. なテストセット上で高い平均精度を示しながら、疑似相関が崩れる非典型的なグループでは一貫して失敗し得る。
2. Group DRO は worst-group training loss を最小化する自然な目的関数である。しかし、過剰パラメータ化モデルでは、訓練損失が全グループで消失するため、naive Group DRO は ERM と差を出しにくい。問題は worst-group training loss ではなく、group-wise generalization gap にある。
3. 正則化は worst-group generalization にとって本質的に重要である。強い $\ell_2$ ペナルティや早期終了によって、モデルが訓練データを完全に fit することを防ぐと、Group DRO が改善した worst-group training performance が test performance に結びつきやすくなる。
4. group adjustment は、グループサイズに応じて残る汎化ギャップの違いを見込む補正である。これは、Group DRO の経験的目的を、より test risk に近い形へ修正する操作として理解できる。
5. Group DRO は importance weighting とは異なる。固定された重みで平均損失を最小化するのではなく、グループ上の adversarial distribution を通じて worst-case risk を直接扱う。非凸設定では、DRO 解がどの importance weighting の解としても得られない場合がある。
6. Algorithm 1 により、Group DRO は通常の SGD に近い計算コストでスケーラブルに最適化できる。したがって、Group DRO は理論上の目的関数にとどまらず、大規模ニューラルネットワークにも適用可能な訓練手続きである。

本論文の意義は、Group DRO をニューラルネットワークへ適用したことだけではない。より重要なのは、過剰パラメータ化領域において、平均汎化と最悪グループ汎化が異なる振る舞いを示すことを明確にした点である。

平均汎化だけを見れば、強い正則化は不要に見える場合がある。しかし、最悪グループ汎化を見ると、正則化は依然として中心的な役割を持つ。この観察は、深層ニューラルネットワークの汎化を平均性能だけで理解することの限界を示している。

したがって、本論文の主張は、単に「Group DRO がロバスト性を改善する」というものではない。正確には、

:::message
Group DRO は worst-group training loss を制御する手法であり、過剰パラメータ化モデルにおいて worst-group test performance を改善するには、正則化によって group-wise generalization gap を制御する必要がある。
:::

という主張である。

これが、原論文のタイトルにある **"On the Importance of Regularization for Worst-Case Generalization"** の中身である。