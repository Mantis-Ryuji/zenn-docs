---
title: "I-JEPA 論文解説：画素の再構成から、意味のある表現の予測へ"
emoji: "🔄"
type: "idea"
topics: ["AI","深層学習"]
published: true
---

## 1. はじめに

画像の一部を隠し、残った部分から予測する。この学習問題は単純だが、「何を正解として予測させるか」によって、モデルに求める理解は変わる。

たとえば、鳥の背中を隠した画像を考える。周囲から鳥の姿勢や身体のつながりを推測できても、羽毛の模様や光の反射まで一意に決められるとは限らない。画像認識に役立つ表現を得たいとき、それらすべての画素を当てることを、同じように要求する必要があるのだろうか。

I-JEPA（Image-based Joint-Embedding Predictive Architecture）は、**見えている領域から、隠れた領域の「画素」ではなく「表現」を予測する**自己教師あり学習手法である。正解となる表現は、学習中の別のencoderが作る。

ただし、予測先をベクトルに変えるだけで、意味を理解するモデルになるわけではない。正解も予測も同じ定数ベクトルになってしまえば、画像の内容を区別できなくても誤差はゼロになる。

本記事では、**I-JEPAがどのような予測課題を通じて、画像認識に役立つ表現を学ぶのか**を解説する。予測対象やマスクの取り方、教師の更新方法を取り上げ、その設計の理由と効果を原論文の実験から見ていく。

@[card](https://arxiv.org/abs/2301.08243)

@[card](https://github.com/facebookresearch/ijepa)

## 2. 自己教師あり学習における予測対象の選択

### 2.1 意味のある表現と評価の関係

画像の表現とは、encoderが画像から計算する特徴ベクトルである。自己教師あり学習では、人が付けたカテゴリラベルの代わりに、画像そのものから作った課題を通じてencoderを学習する。

ここで「意味のある表現」を、単に画像を圧縮した表現と同一視することはできない。圧縮していても、撮影時の色合いばかりを保存し、物体の種類を区別できなければ、物体認識には使いにくい。

I-JEPAの論文が重視するのは、**下流タスクに合わせて学習済みencoderのパラメータを大きく更新しなくても、認識に必要な情報を取り出せること**である。これを確かめる評価方法の一つがlinear probingである。encoderの重みを固定し、その出力の上に線形分類器だけを学習する。これに対してfine-tuningでは、ラベルを用いてencoder自体も更新する。

したがって、線形評価が高いことは、カテゴリを分ける情報が線形に読み出しやすいという根拠になる。一方、それだけで、あらゆる物体関係や因果構造まで理解したことにはならない。本記事でも「意味」は、まずこうした評価で確認できる範囲に置く。

### 2.2 不変性・画素再構成・表現予測の違い

自己教師あり学習の設計を比較するには、**何と何を一致させるか**を見ると分かりやすい。

![Joint-Embedding、生成型、Joint-Embedding Predictiveの三つの構成と、誤差を計算する位置](/images/zenn/ijepa-semantic-prediction/paper-fig2.png)
*図1：自己教師あり学習の三つの構成。赤枠の $D$ は比較対象間の誤差、$z$ は予測を条件づける追加情報を表す。出典：Assran et al.（2023）Figure 2*

図1の左は、二つの入力をそれぞれ表現へ変換し、対応する表現を近づけるJoint-Embedding Architectureである。画像では、同じ画像に異なるcropや色変換を施して複数のviewを作り、その間の一致を学ぶ方法が代表的である。DINO[^dino] などがこの系統に属する。

この学習では、**選んだデータ拡張に対して、表現は不変であるべきだという仮定**をモデルに課している。どの変換に不変であるべきかは、データ拡張の設計によって決まる。画像分類では有効でも、cropによって含まれる物体数が変わった二つのviewを同じ表現へ近づけることが、物体検出にも適しているとは限らない。

中央は、入力信号そのものを再構成する構成である。たとえばMasked Autoencoders（MAE）[^mae] は、画像のパッチを隠し、可視パッチの表現から隠れた画素を再構成する。ただし、**画素を正確に再構成するための特徴が、そのまま物体の識別に役立つとは限らない**。再構成では細かな模様や背景の違いも誤差に入るため、それらを復元する能力を高めることが、物体を識別しやすい表現の学習につながるかを確かめる必要がある。

右がJoint-Embedding Predictive Architecture（JEPA）である。こちらも片方の入力からもう片方を予測するが、誤差を計算する場所は表現空間である。I-JEPAでは追加情報 $z$ に、予測したい領域の位置を用いる。

左の構成では、二つのviewをencoderに通した出力どうしを近づける。右の構成では、**一方の出力をpredictorで変換してから、もう一方の出力と比較する**。I-JEPAでは、見えている領域の表現と位置情報を手がかりに、predictorが隠れた領域の表現を予測する。

たとえば、鳥の胴体が見えていて頭が隠れている場合、胴体の表現から、隠した位置にある頭の表現を予測する。このとき、胴体と頭の表現を同じにする必要はない。**領域ごとの違いを保ちながら、その関係を予測する学習**であるため、局所的な形や配置の情報を表現に残せる可能性がある。

### 2.3 I-JEPAの着眼点と帰納バイアス

data2vec[^data2vec] も、画素ではなく表現を予測する手法である。入力全体を処理した教師encoderの表現を正解とし、入力の一部を隠した側から予測する。

I-JEPAは、この表現空間での予測に、**十分な広がりを持つ複数のtargetと、空間的に分布したcontext**を組み合わせる。targetは予測する領域、contextは予測の手がかりとして見せる領域を指す。マスクの設計によって、何を手がかりに何を予測させるかを調整する。

また、複数のview間の表現を一致させる方法に比べ、**人手で決めたデータ拡張の組み合わせへの依存が小さい**。色変換やぼかしなどについて、「どの変換に不変であるべきか」を事前に仮定する部分を減らしている。公式実装では、入力画像の前処理にrandom resized cropと画素値の正規化を使う。[^transforms]

I-JEPAにも、パッチ分割、位置埋め込み、マスクの形や大きさといった帰納バイアスはある。その設計では、**どの変換を同一視するかという仮定を減らし、どの領域を手がかりに何を予測させるかという課題の設計に重点を置いている**。

## 3. I-JEPAの学習問題と更新方法

### 3.1 全画像から作るターゲットと部分的なコンテキスト

I-JEPAは、context encoder、target encoder、predictorの三つで構成される。いずれもVision Transformer（ViT）[^vit] を使う。ViTは画像を小さなパッチへ分割し、パッチ間のSelf-Attentionによって、各位置の表現へ周囲の情報を取り込むモデルである。

![可視領域を処理するcontext encoder、位置ごとに予測するpredictor、全画像を処理するtarget encoder](/images/zenn/ijepa-semantic-prediction/paper-fig3.png)
*図2：I-JEPAの学習構成。上段は可視パッチからの予測、下段は全画像からの正解生成を表す。同じ色の領域に対応するパッチ表現を比較する。出典：Assran et al.（2023） Figure 3*

前処理後の画像を $y$、パッチ数を $N$、encoderの特徴次元を $d$ とする。target encoderを $f_{\bar\theta}$ と書くと、正解側の表現は

$$
S_y=f_{\bar\theta}(y)
=(s_{y,1},\ldots,s_{y,N}),
\qquad s_{y,j}\in\mathbb R^d
$$

である。$\bar\theta$ はtarget encoderのパラメータを表す。

予測する領域を $M$ 個選び、領域の添字を $i\in\{1,\ldots,M\}$ とする。図2下段の青・赤・黄の枠が、それぞれ一つの予測対象領域に対応する。$i$ 番目の枠内にあるパッチの番号（添字）を集めた集合を $B_i\subseteq\{1,\ldots,N\}$ とする。その領域の正解は、$S_y$ から $j\in B_i$ のベクトルを取り出したものである。

ここで順序が重要になる。**target側では、先に全画像をencoderへ通し、その後で対象位置の出力を選ぶ。** 領域を画像から切り出し、それだけをencoderへ渡す処理ではない。したがって $s_{y,j}$ は、位置 $j$ の画素だけでなく、Self-Attentionを通じた全体の文脈を含みうる。

context側では、前処理後の画像 $y$ のパッチ格子上で、面積比85〜100%の正方形の領域を一つ選ぶ。これはcontext用のマスク領域を決める操作であり、画像を切り出してリサイズする前処理のrandom resized cropとは別である。この領域に含まれるパッチの添字集合を $A$ とすると、実際に見せる集合 $C$ は

$$
C=A\setminus\bigcup_{i=1}^{M}B_i
$$

である。つまり、**どのtargetに含まれるパッチもcontextから除く**。target同士は重なってもよいが、予測に用いる可視パッチとtargetは重ならない。

context encoder $f_\theta$ は、この $C$ に残ったパッチだけを処理する。画像全体の黒塗り部分まで計算するのではなく、可視パッチのトークンを入力するため、encoder側の計算を減らせる。元の画像内の位置は位置埋め込みで保持する。

標準的なマスク設定を整理すると次のようになる。[^method]

| 対象 | 設定 |
| --- | --- |
| target数 $M$ | 4個 |
| 各targetの面積比 | 画像の15〜20% |
| targetの縦横比 | 0.75〜1.5 |
| 除去前のcontext領域 $A$ | 面積比85〜100%、正方形 |
| 最終的なcontext $C$ | $A$ から全targetとの重複部分を除去 |

85〜100%は、**除去前の領域の面積比**である。最終的に画像のほぼ全部を見せるという意味ではない。また、4個のtargetには重なりがあるため、15〜20%を単純に4倍して可視率を求めることもできない。

![原画像、大きな領域からtargetを除いたcontext、四つのtargetのマスク例](/images/zenn/ijepa-semantic-prediction/paper-fig4.png)
*図3：マスクの具体例。左から原画像、context、4個のtarget。暗い部分はその列で使わない領域を表す。出典：Assran et al.（2023）Figure 4*

図3のcontext列では、見える領域が一つの小さな矩形に集中せず、画像内に分布している。I-JEPAの「一つのcontext block」は、大きな矩形から複数のtargetを除いたものであり、最終形状まで一つの穴のない矩形であるとは限らない。

### 3.2 位置を条件とするパッチ表現の予測

context encoderの出力を $S_x$、predictorを $g_\phi$ とする。$\phi$ はpredictorのパラメータである。**predictorもViTで、contextのパッチ表現と予測位置を指定するmask tokenを、一つのトークン列として処理する。**

まず、予測位置を指定するトークンを用意する。$i$ 番目のtargetに含まれる各パッチ位置 $j\in B_i$ に対し、

$$
m_j=m_{\mathrm{mask}}+p_j
$$

とする。$m_{\mathrm{mask}}$ は全位置で共有する学習可能なmask vector、$p_j$ は画像内の位置 $j$ を表す位置埋め込みである。**枠内のパッチ1個につき1個のトークン**を用意し、それぞれの予測位置を指定する。隠れた画素の内容は、このトークンには含まれない。

入力時には、$S_x$ をpredictor用の特徴次元へ射影し、可視パッチの位置埋め込みを加える。そのトークン列の後ろに $\{m_j\}_{j\in B_i}$ を連結し、全体をTransformerのSelf-Attentionで処理する。これにより、mask tokenはcontextの情報を参照しながら、指定された位置の表現を予測する。

Transformerを通した後は、mask tokenに対応する出力だけを取り出し、教師と同じ特徴次元 $d$ へ射影する。この処理までを含めて $g_\phi$ と書くと、$i$ 番目のtargetの予測は、

$$
\hat S_y^{(i)}
=g_\phi\left(S_x\mid\{m_j\}_{j\in B_i}\right)
=\left(\hat s_{y,j}^{(i)}\right)_{j\in B_i}
$$

となる。出力の各ベクトル $\hat s_{y,j}^{(i)}\in\mathbb R^d$ が、教師側の同じ位置の表現 $s_{y,j}$ に対応する。[^predictor]

$\mid$ は**位置トークンによる条件付け**である。同じ $S_x$ を使っても、与える位置トークンの集合を変えると予測する領域が変わる。位置付きmask tokenをcontext表現と同じトークン列へ入れることで、この条件付けを実現している。

図2のpredictorの左右にある□は、次のように対応する。

| 図2の位置 | 対応するトークン・表現 |
| --- | --- |
| predictor左の白（灰色）の□ | context encoderが可視パッチから計算した表現 $S_x$ |
| predictor左の色付き□ | target内の各パッチ位置を指定するトークン $m_j$ |
| predictor右の色付き□ | 各パッチの予測表現 $\hat s_{y,j}^{(i)}$。target領域全体では $\hat S_y^{(i)}$ |

右の色付き□は予測結果であり、図2下段のtarget encoderが作る正解表現と比較する。青・赤・黄は、それぞれ異なる予測領域を表す。同じcontext表現を使い、各targetの位置トークンを与えてpredictorを適用する。原論文の手順では、$M$ 個のtargetに対してこの処理を $M$ 回行う。

上付きの $(i)$ は本記事で補った記号である。同じ位置 $j$ が複数のtargetに含まれる場合でも、一緒に渡す位置トークンの集合が異なるため、予測をブロックごとに区別している。

学習時には、**予測と正解を同じパッチ位置 $j$ で比較し、位置ごとに二乗誤差を計算する**。次節では、それらの誤差を足し合わせて一つの損失にする。集約するのは計算済みの誤差であり、予測表現や教師表現をブロック内で平均するわけではない。

predictorの内部特徴次元は標準で384であり、encoderより幅を狭くしている。たとえばViT-H encoderに対しては12層のpredictorを用いる。位置間の関係をSelf-Attentionで処理しつつ、encoderと同じ幅のネットワークを各targetについて動かす費用を避ける設計である。[^method]

### 3.3 表現の回帰とEMAによるtarget encoderの更新

原論文の目的関数は、predictorが予測したパッチ表現と、target encoderが作った正解表現の二乗誤差である。1画像とそのマスクについて、次のように書ける。

$$
\mathcal L(\theta,\phi;\bar\theta)
=\frac{1}{M}\sum_{i=1}^{M}\sum_{j\in B_i}
\left\|
\hat s_{y,j}^{(i)}-\operatorname{sg}(s_{y,j})
\right\|_2^2.
$$

$\operatorname{sg}$ はstop-gradientを表す。値はそのまま使うが、誤差からtarget encoder側へ勾配を流さない、という意味で本記事では明示している。原式に合わせ、ブロック間は $1/M$ で平均し、ブロック内のパッチについては和を取った。学習では画像とマスクをサンプルし、この損失のバッチ平均から、図2上段のcontext encoderのパラメータ $\theta$ とpredictorのパラメータ $\phi$ を更新する。[^loss]

ここには、別画像を負例として遠ざける対照損失も、画素を再構成する損失もない。target encoderが作る正解表現への予測誤差が、context encoderとpredictorを学習する信号になる。

一方、**図2下段のtarget encoderのパラメータ $\bar\theta$ は、上段のcontext encoderのパラメータ $\theta$ の指数移動平均（EMA）で更新する**。更新回数を $t$、勾配更新後のcontext encoderのパラメータを $\theta_{t+1}$ とすると、

$$
\bar\theta_{t+1}
=\tau_t\bar\theta_t+(1-\tau_t)\theta_{t+1}
$$

である。$\tau_t$ は更新前のtarget encoderのパラメータをどれだけ残すかを決める係数で、論文では0.996から1へ線形に増やす。target encoderのパラメータは、context encoderと同じ値で初期化する。[^method]

この更新によって、context encoderの重みの変化をtarget encoderがゆっくり追いかける。target encoderは、その移動平均の重みを使って正解表現を作る。

:::message
原論文の式は二乗誤差だが、公開実装の `src/train.py` は、target encoderの出力を特徴次元でLayer Normalizationしたうえで `smooth_l1_loss` を使っている。本記事の数理的説明は原論文の二乗誤差に基づく。再実装時には、論文の式と公開コードの設定を区別する必要がある。[^implementation]
:::

事前学習後の評価には、原論文では**target encoder**を使う。事前学習に `[CLS]` tokenは使わず、パッチ出力の平均プーリングから画像表現を作る。predictorは事前学習の予測課題を担うものであり、画像分類の出力器として使うものではない。[^evaluation]

## 4. 意味のある予測を支える条件

### 4.1 物体の形や配置を捉えるための帰納バイアス

I-JEPAの予測課題は、**物体の形や部位の配置を手がかりに、隠れた領域の内容を捉える方向へ学習を促す**と解釈できる。この帰納バイアスは、表現を予測することに加えて、比較的大きな領域を隠し、その位置を指定してパッチごとの表現を予測させる、という組み合わせから生じる。

隠れた領域を予測するときには、見えている物体のどの部分がどこへ続くのか、指定された位置にはどの部位がありそうか、といった関係が手がかりになる。さらに、正解にはtarget encoderが全画像を見て作った表現を用いる。context側には、全体の文脈を踏まえた、その位置の表現を予測することが求められる。

その際、細かな模様や色まで画素として復元することは目標に含めない。誤差を測るのはtarget encoderの表現なので、そこに残らない画素の違いまで当てる必要はない。**物体の構造を捉える情報を予測しつつ、画素の細部を再現する負担を減らす**のが、表現空間を予測先にする狙いである。入力の細部を見ないという制約はなく、予測に役立つ細部は利用できる。

この解釈は、原論文の予測表現の可視化とも整合する。別途学習した生成モデルで予測表現を画像化すると、物体の部位や姿勢に対応する構造が現れ、細かな模様や背景はサンプル間で変動する（第5.2節の図4）。[^visualization]

予測先を表現にする効果は、原論文 Table 7で調べられている。予測対象をtarget encoderの表現にする場合と画素にする場合を比較した実験である。encoderはViT-L/16で、事前学習後に重みを固定し、ImageNetの1%ラベルで線形分類器を学習している。[^ablations]

| 予測対象 | 事前学習epochs | 1%ラベルでの線形評価Top-1（%） |
| --- | ---: | ---: |
| target encoderの出力表現 | 500 | **66.9** |
| 画素 | 800 | 40.7 |

この比較では、表現を予測した設定の方が分類情報を線形に読み出しやすくなっている。ただし、事前学習のepoch数は同一ではなく、この表だけで形や配置の情報を直接検証しているわけでもない。形や配置を捉える帰納バイアスという見方は、予測課題の設計と可視化の観察を合わせた解釈である。

次節では、この予測課題を作るうえで、何を隠し、どの手がかりを残すかというマスク設計が持つ効果を見る。

### 4.2 マスクの大きさと文脈の広がり

教師表現を予測していても、手がかりと正解の組み合わせが悪ければ、認識に役立つ課題にはならない。

小さな欠損なら、近傍の模様の連続性だけで埋められる場合がある。一方、大きく隠しすぎると、見えている領域から正解を推測する手がかりが失われる。I-JEPAは、比較的大きなtargetを複数予測しつつ、画像内の広い範囲にcontextを残すことで、この両側を調整する。

これはマスクの設計意図であり、targetを選ぶ際に物体検出器や物体ラベルを使うわけではない。ランダムに選んだ大きな領域が、必ず一つの物体に対応するわけでもない。

原論文 Table 6は、表現予測を共通にしてマスク戦略を比較している。以下はViT-B/16を300 epochs事前学習し、encoderを固定してImageNetの1%ラベルで線形分類器を学習した結果である。[^ablations]

| マスク戦略 | 予測課題 | 平均可視率 | Top-1精度（%） |
| --- | --- | ---: | ---: |
| multi-block | 広いcontextから、複数の矩形領域を予測 | 25% | **54.2** |
| rasterized | 画像の四分割のうち、一領域から残る三領域を予測 | 25% | 15.5 |
| block | 一つの大きな矩形領域を、残りの領域から予測 | 40% | 20.2 |
| random | ランダムなパッチ集合を、残りのパッチから予測 | 40% | 17.6 |

multi-blockとrasterizedは平均可視率が同じでも大きく異なる。さらに、可視率が40%の二方式より、25%のmulti-blockの方が高い。したがって、「見えるパッチを多くすればよい」という説明では足りない。

ただし、この比較ではtargetの個数、形状、配置、面積などが同時に変わる。差のすべてを、contextの空間的な広がりという一要因だけに帰すこともできない。

そこで付録Cの追加実験を見る。同じViT-B/16・300 epochs・1%線形評価で、targetを4個に固定したとき、面積比の範囲を7.5〜20%から15〜20%へ変えると19.2%から54.2%へ改善する。一方、さらに20〜25%へ大きくすると38.9%へ低下する。**十分に大きいtargetは重要だが、大きいほどよいわけではない。** targetを広げるとcontextから除かれる領域も変わるため、必要なのは、予測の難しさと手がかりの両立である。[^mask-ablation]

文脈の使い方は教師側にも現れる。全画像を教師へ通してから出力を選ぶ場合と、各target領域だけを入力する場合を比較すると、ViT-H/16・300 epochs・1%線形評価で、前者が67.3%、後者が56.1%だった（原論文 Table 11）。[^mask-ablation]

これは第3.1節の処理順を裏づける実験である。I-JEPAが予測するのは、単にその場所だけを圧縮したベクトルではなく、**全画像の中でその場所を見たときの表現**である。予測する側には部分的な観測しか与えず、正解を作る側には全体を見せる。この情報の非対称性も、予測課題の一部になっている。

:::details MSNとの共通点と予測対象の違い
Masked Siamese Networks（MSN）[^msn] も、一部のパッチを隠したviewから、マスクしていないviewの表現を学ぶ手法である。画素を再構成せず、可視パッチだけを処理することや、正解側のencoderをEMAで更新することもI-JEPAと共通する。

MSNでは、各viewから画像単位の表現を作り、学習可能なprototype（代表ベクトル）への割当分布が一致するように学習する。cropや色変換、パッチの欠損によって見え方が変わっても、同じ画像に由来するviewには同じ割当を与えようとする。欠損に対して、画像単位の意味表現をそろえる設計である。

I-JEPAでは、contextと位置トークンから、指定された領域内の各パッチの表現を予測する。同じcontextでも、位置を変えれば別の領域が予測対象になる。contextの表現そのものをtargetの表現と一致させることは要求せず、predictorに位置に応じた対応関係を学ばせる。[^method]

**MSNは欠損に対する画像単位の不変性を学び、I-JEPAは位置に応じた領域間の予測を学ぶ**、という違いがある。部分的な観測から表現を学ぶという発想を共有しつつ、何を一致させるかが異なる。この違いは、第4.1節で述べた、局所的な形や配置を捉える帰納バイアスにも関わると解釈できる。
:::

### 4.3 表現崩壊の回避とpredictorの準最適性

すべての画像で予測も正解も同じ定数ベクトルになれば、損失はゼロになる。損失の式だけを見ると、このような表現崩壊も「正解」に見える。しかし、I-JEPAの一回の更新では、図2下段のtarget encoderが作る正解を固定したまま、上段のcontext encoderとpredictorを学習する。target encoderの重みを動かすのは、その後のEMAである。

この更新を理解する鍵になるのが、BYOL[^byol]で論じられた **predictorの準最適性（near-optimality）** である。準最適とは、**今のcontext表現を使ってできる予測を、predictorがほぼできている状態**を指す。この見方をI-JEPAへ当てはめて考える。

predictorが十分に学習できていても、context encoderが正解の予測に必要な情報を捨ててしまえば、それをpredictorだけで補うことはできない。target encoderが作る正解は固定されているので、context側が情報を減らすだけでは予測誤差は小さくならない。したがって、さらに誤差を減らすには、**context encoderが予測に必要な情報を残す方向へ学習すること**が求められる。

その後、context encoderが学んだ重みを、EMAでtarget encoderへ少しずつ引き継ぐ。正解側が一度に大きく変わると、predictorは変化に追いつけず、十分に学習できた状態を保ちにくい。**EMAは正解をゆっくり変化させ、predictorがその予測に追いつきやすくする**役割を持つと考えられる。BYOL原論文はこの解釈を述べ、I-JEPA原論文もEMAによる更新が重要だったと報告している。[^byol][^loss]

このように、**学習を動的系として捉え、predictorの準最適性を仮定すると、表現崩壊が避けられる理由を説明できる**。

:::details 条件付き分散と情報保持の導出
ここでは第3.3節の、原論文に記載された二乗誤差を用いる。一つのtarget位置での誤差に注目するが、ブロック内の各パッチの誤差にも同じ分解を適用できる。target encoderのパラメータ $\bar\theta$ は、この計算中は固定する。

教師表現を $Z=s_{y,j}\in\mathbb R^d$、予測に使える情報を $U_\theta=(S_x,B_i,j)$ とする。$U_\theta$ にはcontext表現と、位置トークンが指定する領域・パッチ位置を含めている。期待値は画像・マスク・予測位置のサンプリングについて取り、$\mathbb E[\|Z\|_2^2]<\infty$ を仮定する。

その位置に対するpredictorの出力を $a_\phi(U_\theta)$ とし、

$$
R(\theta,\phi)
=\mathbb E\left[\|Z-a_\phi(U_\theta)\|_2^2\right]
$$

と置く。計算を短く書くため、$A=a_\phi(U_\theta)$、$\mu=\mathbb E[Z\mid U_\theta]$ とする。

$$
\begin{aligned}
R(\theta,\phi)
&=\mathbb E[\|Z-A\|_2^2]\\
&=\mathbb E[\|(Z-\mu)+(\mu-A)\|_2^2]\\
&=\mathbb E[\|Z-\mu\|_2^2]
+\mathbb E[\|\mu-A\|_2^2]\\
&\quad+2\mathbb E[(Z-\mu)^\top(\mu-A)].
\end{aligned}
$$

$\mu-A$ は $U_\theta$ の関数なので、交差項は

$$
\begin{aligned}
\mathbb E[(Z-\mu)^\top(\mu-A)]
&=\mathbb E\left[
\mathbb E[Z-\mu\mid U_\theta]^\top(\mu-A)
\right]\\
&=0
\end{aligned}
$$

となる。最後の等号には、$\mathbb E[Z-\mu\mid U_\theta]=\mathbb E[Z\mid U_\theta]-\mu=0$ を使った。したがって、

$$
R(\theta,\phi)
=\mathbb E[\|Z-\mu\|_2^2]
+\mathbb E[\|\mu-a_\phi(U_\theta)\|_2^2]
$$

である。右辺第1項はpredictorの選び方に依存しない。十分な表現能力を持つ理想的なpredictorなら、第2項をゼロにでき、$a_\theta^\star(U_\theta)=\mu$ が最適になる。

この最適予測でも残る誤差を $R_\theta^\star$ とする。$k=1,\ldots,d$ を教師表現の特徴成分の添字とし、ノルムを成分ごとに展開すると、

$$
\begin{aligned}
R_\theta^\star
&=\mathbb E[\|Z-\mu\|_2^2]\\
&=\mathbb E\left[
\sum_{k=1}^{d}(Z_k-\mu_k)^2
\right]\\
&=\mathbb E\left[
\sum_{k=1}^{d}\mathbb E[(Z_k-\mu_k)^2\mid U_\theta]
\right]\\
&=\mathbb E\left[
\sum_{k=1}^{d}\operatorname{Var}(Z_k\mid U_\theta)
\right]
\end{aligned}
$$

となる。また、

$$
R(\theta,\phi)-R_\theta^\star
=\mathbb E\left[
\|a_\phi(U_\theta)-a_\theta^\star(U_\theta)\|_2^2
\right]
\geq 0
$$

が、最適predictorからの超過誤差である。準最適性は、この超過誤差が小さい状態として捉えられる。

次に、contextの情報を捨てた場合を比較する。位置指定は保ったまま、情報を持つ入力 $U$ から、情報を減らした入力 $V=h(U)$ を作るとする。$\mu_U=\mathbb E[Z\mid U]$、$\mu_V=\mathbb E[Z\mid V]$ と置く。$V$ は $U$ の関数なので、$\mu_U-\mu_V$ も $U$ の関数である。このため、

$$
\begin{aligned}
\mathbb E[(Z-\mu_U)^\top(\mu_U-\mu_V)]
&=\mathbb E\left[
\mathbb E[Z-\mu_U\mid U]^\top(\mu_U-\mu_V)
\right]\\
&=0
\end{aligned}
$$

となる。この交差項を使って誤差を分解すると、

$$
\begin{aligned}
R_V^\star
&=\mathbb E[\|Z-\mu_V\|_2^2]\\
&=\mathbb E[\|(Z-\mu_U)+(\mu_U-\mu_V)\|_2^2]\\
&=R_U^\star+\mathbb E[\|\mu_U-\mu_V\|_2^2]\\
&\geq R_U^\star
\end{aligned}
$$

を得る。**入力から情報を捨てるだけでは、最適predictorでも残る誤差を減らせない。** 捨てた情報によって教師の条件付き平均が変わる場合、右辺第2項は正になる。これが、予測に使えていた情報を失う方向に抵抗が生じる、という説明の根拠である。予測に関係しない情報を捨てた場合には、第2項がゼロのままでもよい。

この最適値の議論をcontext encoderの更新方向へつなげるには、追加の仮定が必要になる。条件付き平均を実現する最適predictorのパラメータを、$\theta$ の関数 $\phi^\star(\theta)$ として選べるとする。必要な微分可能性、期待値と微分の交換、最適点の一次条件を仮定すると、連鎖律から

$$
\begin{aligned}
\nabla_\theta R_\theta^\star
&=\left.\nabla_\theta R(\theta,\phi)\right|_{\phi=\phi^\star(\theta)}\\
&\quad+
\left(\frac{\partial\phi^\star(\theta)}{\partial\theta}\right)^\top
\underbrace{\nabla_\phi R(\theta,\phi^\star(\theta))}_{=0}\\
&=\left.\nabla_\theta R(\theta,\phi)\right|_{\phi=\phi^\star(\theta)}
\end{aligned}
$$

となる。$\partial\phi^\star(\theta)/\partial\theta$ は、最適predictorのパラメータがcontext encoderの変化にどう依存するかを表すJacobianである。predictorが最適なので、その変化を介する項は一次条件によって消える。残るのは、predictorのパラメータを固定し、その入力を通してcontext encoderへ逆伝播する勾配である。これが、最適predictorの下ではcontext側の更新を、期待条件付き分散を下げる更新として読める理由になる。

この等式はpredictorが厳密に最適な場合のものである。準最適なpredictorで同じ更新方向を近似すると読むには、近似誤差が勾配も大きく変えないことが必要になる。小さい予測誤差だけで、更新方向の近さや学習全体の安定性まで保証されるわけではない。

また、**崩壊した不動点が存在することと、その不動点が周囲の状態を引きつけること**は別である。両encoderが既に同じ定数表現になった状態は、不動点として残りうる。BYOL原論文も、最適predictorの下でこうした平衡点が不安定になるという見方を仮説として提示している。ここでは、その情報保持の議論をI-JEPAへ適用した解釈を述べており、ViTやEMAを含む学習全体の安定性を証明したものではない。
:::

BYOLでの導出と更新ダイナミクスについては、[BYOLの解説記事](https://zenn.dev/mantis_ryuji/articles/c8076edeeae3b0)でも説明している。

## 5. 実験結果と評価の限界

### 5.1 線形評価が示す分類表現の読み出しやすさ

**I-JEPAは、ImageNet-1Kの線形評価でMAEを上回り、encoderを追加学習しなくても分類情報を取り出しやすい表現を獲得している。** この点を、学習済みencoderの重みを固定し、その出力の上に線形分類器だけを学習する評価で確かめる。

次表は原論文 Table 1から必要な行を抜粋したものだ。事前学習にはImageNet-1Kの画像をラベルなしで使い、線形分類器の学習には**全学習ラベル**を使う。epoch数は事前学習の回数である。[^classification]

| 手法 | encoder | 事前学習epochs | Top-1精度（%） |
| --- | --- | ---: | ---: |
| MAE | ViT-H/14 | 1600 | 77.2 |
| I-JEPA | ViT-H/14 | 300 | **79.3** |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| I-JEPA | ViT-L/16 | 600 | 77.5 |
| iBOT[^ibot] | ViT-L/16 | 250 | 81.0 |
| I-JEPA | ViT-H/16（448解像度） | 300 | 81.1 |

ViTのB・L・HはそれぞれBase・Large・Hugeを表し、`/14` や `/16` はパッチの一辺の画素数を表す。本論文のI-JEPAは原則224 × 224で事前学習し、表の最後の行は448 × 448を使う。

中心となるのは、同じViT-H/14を使ったMAEとの比較である。MAEは1600 epochsの事前学習で77.2%、I-JEPAは300 epochsで79.3%だった。**encoderを更新しない条件で、I-JEPAは少ない事前学習epochsで2.1ポイント高い精度を得ている。** これは、I-JEPAが事前学習の段階で、分類に必要な違いを線形に取り出しやすい表現を作っていることを支持する。

data2vecとのViT-L/16の比較では、精度は77.3%と77.5%で近いが、事前学習はそれぞれ1600 epochsと600 epochsである。ここでもI-JEPAは、少ない事前学習epochsで同程度の線形評価の精度を得ている。

さらに、高解像度のI-JEPAは81.1%に達し、複数view間の不変性を学ぶiBOTの81.0%に並ぶ。こちらはモデルサイズと解像度が異なる比較だが、I-JEPAでもこの水準の線形分類性能に到達できることを示している。

:::details ラベル数とfine-tuningによる評価条件の違い
ラベルを1%に減らした比較では、ViT-H/14のI-JEPAは73.3%、MAEは71.5%だった。これは原論文 Table 2の評価であり、各手法についてfine-tuningまたはlinear probingのよい方を採用している。I-JEPAは50 epochsのfine-tuningを行う。**第4章のアブレーションで使った「1%ラベルの線形評価」とは、encoderを更新するかどうかが違う。**[^classification]

全ラベルでfine-tuningすると、448 × 448でI-JEPAのViT-H/16が87.1%、MAEのViT-H/14が87.8%である。事前学習はそれぞれ300、1600 epochsで、パッチサイズも異なる。線形評価での優劣が、そのまま維持されるわけではない。[^finetuning]

線形評価は、事前学習で得た表現から分類情報を取り出しやすいかを調べる。fine-tuningは、ラベルでencoderも調整した後の性能を調べる。両者を分けることで、事前学習した表現から線形に読み出せる情報と、教師ありの追加学習後の分類性能を区別できる。
:::

**encoderを固定したまま高い分類性能を得られることが、I-JEPAの表現の有用性を支えている。** 次節では、分類情報に加えて、物体数や距離などの情報も読み出せるかを見る。

### 5.2 分類情報と画像内の構造に関する情報の両立

**I-JEPAは、分類情報を読み出しやすい表現を得ると同時に、物体数や距離などの情報も保持している。** encoderを固定した評価と、predictorの予測表現の可視化から、この特徴を見る。

原論文 Table 4では、encoderを固定し、CLEVRの物体数と距離に関する課題を線形に読み出している。CLEVR/Countは物体数の分類、CLEVR/Distは最も近い物体までの距離を区間に分ける分類である。以下は精度（%）の抜粋である。[^local][^clevr]

| 手法 | encoder | CLEVR/Count | CLEVR/Dist |
| --- | --- | ---: | ---: |
| MAE | ViT-H/14 | **90.5** | **72.4** |
| I-JEPA | ViT-H/14 | 86.7 | **72.4** |
| DINO | ViT-B/8 | 86.6 | 53.4 |
| iBOT | ViT-L/16 | 85.7 | 62.8 |

表の設定では、I-JEPAの距離分類は72.4%で、DINOの53.4%、iBOTの62.8%を上回り、MAEと同じ精度を得ている。第5.1節の結果と合わせると、**MAEより高い線形分類精度を得ながら、距離情報もMAEと同じ水準で読み出せている**。一方、物体数ではMAEが上回り、DINOとの差も0.1ポイントである。I-JEPAの特徴は、分類情報とこうした構造に関する情報が両方使える点にある。

predictorの予測表現を調べるため、原論文ではRCDM（Representation-Conditioned Diffusion Model）[^rcdm] による可視化も行っている。これは、**別途学習した拡散decoderに、predictorの出力を条件として与えて画像を生成する方法**である。[^visualization]

![鳥、車、犬などのcontextと、予測表現を条件に別の生成モデルが画像化した複数サンプル](/images/zenn/ijepa-semantic-prediction/paper-fig6.png)
*図4：predictorの出力表現の可視化。左右それぞれの例で、左から元画像、context、4個の生成サンプルを並べている。緑枠が予測対象領域。出典：Assran et al.（2023）Figure 6*

鳥の例では背中に対応する領域に鳥の身体らしい構造が、車の例では車体の上部に対応する構造が現れる。一方、細部の模様や色合いはサンプル間で変動する。共通して現れる部位や姿勢は、predictorがその位置の物体構造を捉えた表現を予測していることを示唆する。

この結果は、第4.1節で述べた、**物体の形や配置を捉えながら、画素の細部を再現する負担を減らす**という設計意図を支持する観察である。

:::details 可視化の手順と読み取りの範囲
まず、学習済みのcontext encoderとpredictorを固定する。次に、predictorのパッチ出力を平均プーリングした表現を条件として、画像へ戻す拡散decoderを別途学習する。図4の緑枠内は、そのdecoderへ異なる乱数を与えて得たサンプルである。可視化の対象は、target encoderそのものではなくpredictorの予測表現である。

このdecoderはI-JEPAの事前学習には含まれない。また、表示される画像にはdecoderの能力や事前知識も反映される。サンプル間で変わる情報が元の表現に存在しないと証明したものではなく、予測表現に残る構造を探るための定性的な手がかりとして読む。
:::

### 5.3 反復回数の削減による事前学習の効率化

**I-JEPAは、有用な表現を得るまでの反復回数を減らすことで、事前学習全体の計算コストを抑えている。**

target encoderが全画像を処理する追加コストがあるため、原論文のMAEとの比較では、1 iterationあたりの処理時間は約7%長い。一方、必要なiterationは約5分の1だった。**一回の処理にかかる追加コストより、必要な反復回数が減る効果の方が大きく、総学習コストが下がっている。**[^compute]

構成上も、context encoderには可視パッチだけを渡し、predictorには幅の狭いViTを使う。複数viewを処理する手法に対しては、target encoderが一つのviewだけを処理すればよい点も効率につながる。論文では、ImageNet上のViT-H/14を16台のA100で72時間未満で事前学習できたと報告している。[^compute]

第5.1節の、少ない事前学習epochsで高い線形分類精度を得る結果と合わせると、**何を予測させるかという課題の設計は、有用な表現を得るまでの計算コストにも関わる**ことが分かる。

:::details 計算量の比較条件と適用範囲
学習時間は、論文で使われたモデル・データ・実装における結果である。複数viewを使う手法では、1画像あたりに処理する入力の数も異なる。epoch数、1 stepの処理時間、GPU台数を掛けたGPU-hours、実時間は、それぞれ区別して比較する必要がある。

また、計算効率の改善が、すべてのタスクでの優位性を意味するわけではない。転移分類のiNaturalist18では、線形評価がI-JEPAのViT-H/14で47.6%、DINOのViT-B/8で55.9%、iBOTのViT-L/16で57.3%だった。[^classification]

I-JEPAは、異なる領域の間に予測に使える依存関係があることを利用する。たとえば、周囲から予測しにくい微小な異常が重要な課題では、大きなブロックの予測と抽象化が適切かどうかを別途評価する必要がある。これは本論文が実験した失敗例ではなく、予測課題と下流で必要な情報の違いから生じる検討事項である。

本論文の対象は静止画像内の領域間の関係であり、時間的な変化、行動の結果、因果介入を学習・評価したものではない。
:::

## 6. さいごに

I-JEPAから持ち帰りたいのは、**データの構造を手がかりに、下流で必要な情報を学べる予測課題（pretext task）を考える**という視点である。

I-JEPAの設計が促すのは、**物体の形や配置を捉える抽象化**だと解釈できる。見えている胴体から隠れた頭の表現を予測するときには、部位ごとの違いを保ちながら、そのつながりや配置を捉えることが手がかりになる。正解を表現空間に置くことで、そこに含まれない画素の細部を再現する負担も減らせる。予測する対象と与える文脈を選ぶことが、こうした抽象化を促す帰納バイアスになる。

この個性は、事前学習済みcheckpointを選ぶときの参考にもなる。第5章で見た分類情報と物体数・距離の情報の両立は、I-JEPAを選ぶ理由の一つになりうる。何を予測する課題で学んだかを知ることは、その表現が自分の用途に合うかを考える手がかりになる。

自分で事前学習を行う場合には、この対応を予測課題の設計に反映することになる。I-JEPAが自然画像にある部位のつながりや配置を利用したように、別のドメインでも、そのデータの成り立ちや構造から予測に使える関係を考えたい。**その関係を学ぶことで下流に必要な情報が残るよう、データの背景に合ったpretext taskを設計することが大切である。**

[^dino]: Mathilde Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021。DINOは、異なるviewに対する自己蒸留を用いる。
[^mae]: Kaiming He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022。
[^data2vec]: Alexei Baevski et al., [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555), ICML 2022。全入力から得る文脈を含む教師表現を、マスクされた入力から予測する。
[^msn]: Mahmoud Assran et al., [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141), ECCV 2022。[第3節](https://arxiv.org/html/2204.07141v1#S3)に、viewのマスク、prototypeへの割当分布の整合、target encoderのEMA更新を記載。
[^byol]: Jean-Bastien Grill et al., [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733), NeurIPS 2020。第3.2節では最適predictorと条件付き分散から崩壊点の不安定性を仮説として説明し、付録Hで更新方向との関係、付録Iで準最適性に関する実験を示す。
[^transforms]: 公式実装の[画像変換](https://github.com/facebookresearch/ijepa/blob/main/src/transforms.py)と[ViT-H/14の標準設定](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml)。この設定ではcolor distortion、Gaussian blur、horizontal flipは無効だが、random resized cropは用いる。
[^vit]: Alexey Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021。
[^method]: I-JEPA原論文[第3節](https://arxiv.org/html/2301.08243v3#S3)および[付録A.1](https://arxiv.org/html/2301.08243v3#A1.SS1)。ミニバッチ処理では、同じGPU上で扱うcontextのパッチ数、targetのパッチ数をそれぞれ揃える処理も行う。
[^predictor]: 公式実装の[`VisionTransformerPredictor`](https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py#L200-L300)。`forward`でトークンの連結、Transformerによる処理、予測出力の取り出しを確認できる。
[^loss]: I-JEPA原論文[第3節「Loss」](https://arxiv.org/html/2301.08243v3#S3)。二乗ノルムをブロック内で加算する原式と、教師をEMAで更新する手順に基づく。
[^implementation]: 公式実装の[`src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py)。`forward_target`では勾配計算を止めた教師出力に`layer_norm`を適用し、`loss_fn`では`smooth_l1_loss`を使う。
[^evaluation]: I-JEPA原論文[付録A](https://arxiv.org/html/2301.08243v3#A1)。下流の線形評価では、最終層の平均プーリングだけでなく、最後の4層をそれぞれ平均プーリングして連結した表現や、分類head前のBatch Normalizationの有無も比較し、よい設定を採用する。
[^ablations]: I-JEPA原論文[第9節とTables 6–7](https://arxiv.org/html/2301.08243v3#S9)。両実験とも1%ラベルの線形評価だが、マスク戦略はViT-B/16、予測空間はViT-L/16の実験である。
[^mask-ablation]: I-JEPA原論文[付録C、Tables 8–11](https://arxiv.org/html/2301.08243v3#A3)。target面積、context面積、target個数、教師へのマスク位置を比較している。
[^classification]: I-JEPA原論文[第5節、Tables 1–3](https://arxiv.org/html/2301.08243v3#S5)および[付録A.2](https://arxiv.org/html/2301.08243v3#A1.SS2)。Table 1は全ラベルの線形評価、Table 2は1%ラベルで各手法に適した適応方法を採用した比較、Table 3は転移先での線形評価である。
[^ibot]: Jinghao Zhou et al., [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832), ICLR 2022。異なるview間の自己蒸留とパッチ単位の学習を組み合わせる。
[^finetuning]: I-JEPA原論文[付録D、Table 15](https://arxiv.org/html/2301.08243v3#A4)。I-JEPAでは全ImageNetラベルで50 epochsのfine-tuningを実施する。
[^local]: I-JEPA原論文[第6節、Table 4](https://arxiv.org/html/2301.08243v3#S6)。評価時の入力変換も付録A.2に記載され、CLEVRでは物体を除いてしまうrandom cropを避けている。
[^clevr]: VTABの[公式CLEVR実装](https://github.com/google-research/task_adaptation/blob/master/task_adaptation/data/clevr.py)では、Countは3〜10個の8クラス、Distは最近傍物体の距離を6区間に分けた分類として定義されている。
[^rcdm]: Florian Bordes, Randall Balestriero, Pascal Vincent, [High Fidelity Visualization of What Your Self-Supervised Representation Knows About](https://openreview.net/forum?id=urfWb7VjmL), TMLR 2022。
[^visualization]: I-JEPA原論文[第8節](https://arxiv.org/html/2301.08243v3#S8)および[付録E](https://arxiv.org/html/2301.08243v3#A5)。predictor出力を平均プーリングし、別途学習したRCDM decoderの条件として可視化する。
[^compute]: I-JEPA原論文[第7節](https://arxiv.org/html/2301.08243v3#S7)およびAbstract。所要時間は論文の実験報告であり、本記事で事前学習を追試した結果ではない。
