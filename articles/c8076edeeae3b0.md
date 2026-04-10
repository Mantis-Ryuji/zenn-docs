---
title: "BYOLを理解する：負例なし学習と崩壊回避のメカニズム"
emoji: "🐈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: []
published: false
---

## BYOLを理解する：負例なし学習と崩壊回避のメカニズム

自己教師あり学習では、長いあいだ、 **同じ画像から得た2つの view は近づけ、異なる画像から得た view は遠ざける** という考え方が中心にあった。SimCLR[^1] や MoCo[^2] に代表される contrastive learning は、この枠組みのもとで大きな成功を収めた。しかしその一方で、その性能は「負例をどのように集めるか」「大きなバッチサイズをどう確保するか」「どの augmentation を組み合わせるか」といった設計に強く依存していた。BYOL は、まさにこの常識に切り込んだ手法である。論文は、 **負例をまったく用いずに** 高い性能を達成できること、さらに contrastive learning と比べて augmentation や batch size の変更に対して頑健であることを示した。実際、ResNet-50 の線形評価では top-1 accuracy 74.3%、より大規模な ResNet では 79.6% に到達している。

![](https://storage.googleapis.com/zenn-user-upload/35b82026023c-20260410.png)
*Fig1. ImageNet における BYOL の性能（線形評価）。ResNet-50 および ResNet200（2×）を用いた結果を、他の教師なし手法および教師あり（Sup.）ベースラインと比較したもの。*


ただし、BYOL の本当の面白さは、単に「負例が不要だった」という点にはない。むしろ本質的な問いは、**なぜそれで崩壊しないのか** という点にある。表現空間で単に「一致」だけを学習するなら、すべての入力を同じ定数ベクトルへ写してしまう **自明解** に落ちてもおかしくない。それにもかかわらず、BYOL は実験的に崩壊しない。しかもそれは偶然ではなく、 **online network / target network / predictor / stop-gradient / EMA** からなる非対称な更新ダイナミクスのうえで成立している。したがって、BYOL を理解するとは、単に損失関数の形を追うことではなく、この更新ダイナミクスを理解することである。

本記事では、まず BYOL 以前の自己教師あり学習の文脈を簡潔に振り返り、そのうえで BYOL の目的関数と更新則を数式とともに検討する。とくに焦点となるのは、論文が示す「これは単純な joint minimization ではない」という視点と、predictor の **near-optimality** [^3]および条件付き分散から理解される崩壊回避の直観である。以下では、BYOL を単なる *negative-free SSL* ではなく、 **自己予測を安定化するための設計原理** として読んでいく。

[^1]: [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)
[^2]: [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722)
[^3]: near-optimality : ここでは、predictor が各時刻で厳密な最適解そのものに到達していることではなく、少なくともその最適解に十分近い状態にあることを指す。BYOL の理論的な議論では、この仮定のもとで predictor の振る舞いを単純化し、崩壊回避の機構を理解しやすくしている。

@[card](https://arxiv.org/pdf/2006.07733)
@[card](https://github.com/google-deepmind/deepmind-research/tree/master/byol)

---

## 1. BYOL以前の自己教師あり学習

### 1.1 contrastive learning の基本図式

自己教師あり学習の目的は、ラベルなしデータから下流タスクに有用な表現を学習することである。BYOL 論文が書かれた時点では、その中心にあったのは contrastive learning だった。基本的なアイデアは、同一画像から得た 2 つの拡張 view を **正例** として近づけ、異なる画像由来の view を **負例** として遠ざける、というものである。これにより、モデルは「同じ意味内容を保つ変動」には不変でありつつ、「異なる画像」は区別できる表現を獲得する。

この枠組みは強力だったが、負例の扱いには繊細な設計が求められた。実際、当時の最先端手法は、大きなバッチサイズやメモリバンク、あるいは負例を効果的に利用するための専用の工夫に支えられていた。さらに、その性能は augmentation の選び方にも大きく左右された。要するに contrastive learning は、単に正例と負例を与えれば成立するものではない。重要なのは、 **モデルにとって十分に難しく、しかも意味のある識別問題をいかに構成するか** にあった。

### 1.2 崩壊とは何か

ここで、表現学習における崩壊を明示しておく。入力画像空間を $\mathcal{X}$ 、表現空間を $\mathbb{R}^d$ とし、表現写像を

$$
f_\theta:\mathcal{X}\to\mathbb{R}^d
$$

とする。崩壊とは、たとえば

$$
f_\theta(x)=c \qquad \forall x\in \mathcal{X}
$$

となるような、 **入力に依らず同じ表現しか出さない状態** を指す。こうした表現は、下流タスクに必要な情報をほとんど保持していないため、表現学習としては失敗である。

なぜ崩壊が問題になるのか。cross-view prediction を表現空間でそのまま行うと、 **すべての入力を同じ定数ベクトルへ写す解** でも目的を達成できてしまうからである。論文でも、表現空間で直接予測を行う場合には、「view に依存しない一定の表現は、常に自分自身を完全に予測できる」と指摘されている。contrastive learning は、この問題を **識別問題へと組み替える** ことで回避していた。つまり、同じ画像の別 view を対応づけるだけでなく、他の画像から得られた view とも区別させることで、定数表現では解けない課題へと変換していたのである。

### 1.3 BYOL の問題設定

この文脈において、BYOL の問題設定は明快である。

$$
\textbf{「高い表現性能と崩壊回避を両立するうえで、負例は本当に不可欠なのか。」}
$$

BYOL は、負例こそが自己教師あり学習の本質なのではなく、 **安定したターゲットを用いた自己予測** という別の原理によっても有効な表現学習が可能ではないか、という立場から出発する。従来のブートストラップ的手法[^4]では、疑似ラベルやクラスタ ID のような、表現そのものではない補助的な目標を足場としていた。これに対して BYOL は、表現それ自体をブートストラップの対象とする。すなわち、ある時点の表現を予測対象として定め、別の view から得られた表現がそれを予測するように学習する。さらに重要なのは、そのターゲットが静的に固定されたものではなく、online network の重みを指数移動平均で追従する target network によって与えられる点である。BYOL の要点は、この非対称なターゲット設計にある。

[^4]: ブートストラップ的手法: 自己教師あり学習において、モデル自身が生成した信号を次の学習の目標として再利用する枠組みを広く指す。典型的には、疑似ラベル、クラスタ ID、過去時刻の予測結果などが用いられる。BYOL はこの系譜に属するが、その特徴は、外在的な離散ラベルではなく表現ベクトルそのものをブートストラップする点にある。

---

## 2. BYOLの全体像

### 2.1 online network と target network

BYOL は 2 本のネットワークを用いる。1 つは学習される **online network** 、もう 1 つはそのゆっくり移動するコピーである **target network** である。online 側はパラメータ $\theta$ を持ち、target 側は $\xi$ を持つ。重要なのは、target 側は通常の勾配降下では更新されず、online 側のパラメータを指数移動平均で追従する、という点である。論文では、各ステップ後に

$$
\xi \leftarrow \tau \xi + (1-\tau)\theta
\tag{1}
$$

と更新される。ここで $\tau\in[0,1]$ は target decay であり、BYOL では学習中にこの値を高めていく。

この構造により、target network は「online の少し過去の、より滑らかなバージョン」として機能する。論文の表現を借りれば、BYOL は固定チェックポイントを使ってブートストラップするのではなく、 **online network のゆっくり動く指数平均をターゲットとして、表現を反復的に洗練する** 。この「少し遅れた自分」を教師として使う構図が BYOL の中核にある。

### 2.2 predictor を含む非対称構造

BYOL の online network は 3 つの写像から構成される。

* encoder $f_\theta$
* projector $g_\theta$
* predictor $q_\theta$

一方、target network は

* encoder $f_\xi$
* projector $g_\xi$

のみを持つ。つまり、**predictor は online 側にしか存在しない** 。この非対称性が重要である。論文でも明示されているように、online 経路と target 経路は対称ではなく、predictor を片側のみに置くことが崩壊回避に本質的な役割を果たしていると考えられている。

入力画像 $x$ に対して、2つの augmentation 分布 $\mathcal{T}, \mathcal{T}'$ からサンプルした変換 $t,\ t'$ を用いて、 $v=t(x),\ v'=t'(x)$ という2つの view を生成する。BYOL では、一方の view を online network に、もう一方を target network に入力し、online 側が target 側の射影表現を予測するように学習を行う。この構図は一見すると単純だが、 **予測方向、勾配の流れ、パラメータ更新の仕方** が非対称に設計されている点で、単なる対称な一致学習とは本質的に異なる。

![](https://storage.googleapis.com/zenn-user-upload/a160597e4808-20260410.png)
*Fig2. BYOL のアーキテクチャ。BYOL は、$q_\theta(z_\theta)$ と $\mathrm{sg}(z_\xi^\prime)$ の間の類似度損失を最小化する。ここで、$\theta$ は学習される重み、$\xi$ は $\theta$ の指数移動平均、$\mathrm{sg}$ は stop-gradient を意味する。学習の終了時には、$f_\theta$ 以外のすべては破棄され、$y_\theta$ が画像表現として用いられる。*

---

## 3. 目的関数と更新則

### 3.1 BYOL の基本計算

まず、BYOL における online network、target network、predictor の計算を定義する。online 側では、view $v$ に対して

$$
y_\theta = f_\theta(v), \qquad z_\theta = g_\theta(y_\theta)
$$

を計算する。target 側では、別 view $v'$ に対して

$$
y'_\xi = f_\xi(v'), \qquad z'_\xi = g_\xi(y'_\xi)
$$

を得る。さらに online 側のみ predictor を通して

$$
p_\theta = q_\theta(z_\theta)
$$

を計算する。論文の記法では、これらを $\ell_2$ 正規化した

$$
\bar p_\theta = \frac{p_\theta}{\|p_\theta\|_2}, \qquad \bar z'_\xi = \frac{z'_\xi}{\|z'_\xi\|_2}
$$

を用いて損失を定義する。

### 3.2 BYOL 損失

BYOL の片方向損失は

$$
\mathcal{L}_{\theta,\xi}=\left\|\bar p_\theta - \bar z'_\xi\right\|_2^2\tag{2}
$$

である。論文ではこれを展開して、

$$
\mathcal{L}_{\theta,\xi}=2 - 2\cdot\frac{\langle p_\theta, z'_\xi\rangle}{\|p_\theta\|_2\|z'_\xi\|_2}
$$

と書いている。したがって、BYOL の損失は本質的には **cosine similarity を最大化する形** になっている。ここで重要なのは、online 側が単に別 view の表現を追いかけるのではなく、**EMA によって更新される target network の表現** を予測対象としている点である。

さらに論文では、 $v$ と $v'$ の役割を入れ替えた逆向きの損失 $\tilde{\mathcal{L}}_{\theta,\xi}$ も計算し、

$$
\mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}=\mathcal{L}_{\theta,\xi}+\tilde{\mathcal{L}}_{\theta,\xi}
$$

として両方向の損失を足し合わせる。このとき、**重み $\theta,\xi$ は同一のままで、online 側と target 側に与える view の割当だけを入れ替える** 。これにより、2つの view はそれぞれ online と target の両方の役割を経験する。

### 3.3 stop-gradient と動的系としての BYOL

各学習ステップで損失に対して最適化されるのは、online 側のパラメータ $\theta$ のみである。

$$
\theta \leftarrow \mathrm{optimizer}\left(\theta,\nabla_\theta \mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi},\eta\right)\tag{3}
$$

ここで $\eta$ は学習率である。

一方、target 側のパラメータ $\xi$ は勾配によっては更新されず、stop-gradient のもとで online 側の重みを指数移動平均により追従する。したがって BYOL には、

* $\theta$ ：損失勾配による更新
* $\xi$ ： $\theta$ の EMA による更新

という、更新則の異なる2つの枝が存在する。

この非対称性は、BYOL の理解において決定的に重要である。表面的には、BYOL の損失は2つの表現を近づけるだけに見えるため、崩壊解もまた目的関数を満たしうるように思われる。しかし実際には、BYOL は $(\theta,\xi)$ を同時に損失最小化する静的な最適化問題としては定義されていない。論文でも、target パラメータ $\xi$ の更新は $\nabla_\xi \mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}$ に沿うものではない以上、その学習ダイナミクスを単純な joint gradient descent とみなすべきではないと指摘している。

したがって、BYOL を理解するには、それを単なる損失最小化としてではなく、 **勾配更新と EMA 更新が結びついた非対称な更新ダイナミクス** として捉える必要がある。崩壊回避の鍵も、この非対称な更新ダイナミクスにある。

---

## 4. なぜ BYOL 崩壊ないのか

### 4.1 一見すると、BYOL は崩壊してもおかしくない

BYOL の損失は、表面的にはきわめて単純である。online network が出した予測 $p_\theta$ を、target network の射影 $z'_\xi$ に近づけるだけだ。もしここで「似せること」だけが目的なら、すべての入力に対して同じ定数ベクトルを出力する表現でも、見かけ上は損失を小さくできてしまうように思える。

実際、定数ベクトル $c\in\mathbb{R}^d$ を用いて

$$
p_\theta(x)=c,\qquad z_\xi(x)=c \qquad \forall x
$$

となれば、正規化後のベクトルどうしも一致し、少なくとも直感的には崩壊解が候補に見える。contrastive learning が負例を導入していたのは、まさにこのような自明解を識別問題によって排除するためだった。したがって、 **負例なしで学習する BYOL がなぜ崩壊しないのか** は、論文の中心的な論点そのものである。BYOL 論文も、明示的な anti-崩壊項を持たないにもかかわらず崩壊しないことを、まず問題として正面から扱っている。

「実験でうまくいったから大丈夫」ではなく、

* なぜ一見、崩壊しそうに見えるのか
* それでもなぜ実際には崩壊しにくいのか
* その説明はどこまで厳密で、どこからが仮説なのか

を順に切り分ける必要がある。

### 4.2 論文の第一の主張：これは単純な joint minimization ではない

BYOL の理解でまず重要なのは、これを単純に

$$
\min_{\theta,\xi}\mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}
$$

という静的最適化問題として読むのは不正確だ、という点である。
確かに online 側の $\theta$ は損失 $\mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}$ を小さくするように勾配降下で更新される。しかし target 側の $\xi$ は、その損失に関する勾配 $\nabla_\xi \mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}$ に従って動くわけではない。実際には

$$
\xi \leftarrow \tau \xi + (1-\tau)\theta
$$

という指数移動平均で更新される。つまり、 $\theta$ と $\xi$ は **異なる法則で時間発展する** 。論文はこの点を強調し、BYOL のダイナミクスを $(\theta,\xi)$ に対する同時な勾配降下として記述できる単一の損失 $L^\ast_{\theta,\xi}$ が、そもそも存在しないのではないかと述べている。さらにその直感を、GAN のような非対称な動的系になぞらえている。

この観点は非常に重要である。
なぜなら、崩壊解が損失の最小値になりうる」ことと、「実際の学習ダイナミクスがそこへ収束する」ことは別だからである。静的な目的関数の地形だけを見て BYOL を理解しようとすると、崩壊解がありそう $\rightarrow$ だから学習もそこへ行くはず と考えたくなる。だが BYOL では、target 側が勾配で最適化されない以上、その推論は成立しない。論文の第一のポイントは、 **BYOL をひとつの損失の最小化問題として読むべきではなく、非対称な更新則を持つ動的系として読むべきだ** ということである。

了解。固定値の導入は外して、条件付き期待値と直交分解だけでそのまま導出する形に整理する。式の壊れていた箇所もすべて直した全文を以下に示す。

---

### 4.3 論文の第二の主張：最適 predictor を仮定すると、条件付き分散の最小化として読める

ただし、「BYOL は非対称な更新ダイナミクスを持つから、単純な joint minimization ではない」と言うだけでは、まだ十分ではない。それだけでは、**なぜその更新が崩壊ではなく、有意味な表現学習の方向に働くのか** がなお見えにくいからである。

そこで論文は、さらに一歩踏み込んだ見方を与える。それは、predictor がその時点で十分に最適に近いと仮定すると、BYOL の更新を **target 表現の条件付き分散を小さくする方向** として解釈できる、というものである。

まず、online 側の射影表現を $z_\theta$、target 側の射影表現を $z'_\xi$ とする。predictor $q_\theta$ は、$z_\theta$ を入力として $z'_\xi$ を予測する役割を担っている。このとき本質だけを取り出せば、predictor の学習は

$$
\min_q \mathbb{E}\Bigl[\|q(z_\theta)-z'_\xi\|_2^2\Bigr]
$$

という二乗誤差最小化の問題として書ける。
ここで期待値は、データの分布と augmentation のランダム性に関して取られている。

この目的から最適 predictor の形を導くために、

$$
\mu(z_\theta):=\mathbb{E}[z'_\xi \mid z_\theta]\quad \dots \quad (\ast)
$$

とおく。すると

$$
q(z_\theta)-z'_\xi=\bigl(q(z_\theta)-\mu(z_\theta)\bigr)+\bigl(\mu(z_\theta)-z'_\xi\bigr)
$$

と書ける。したがって、ノルム二乗を展開すると

$$
\|q(z_\theta)-z'_\xi\|_2^2=\|q(z_\theta)-\mu(z_\theta)\|_2^2+\|\mu(z_\theta)-z'_\xi\|_2^2+2\Bigl\langle q(z_\theta)-\mu(z_\theta),\mu(z_\theta)-z'_\xi\Bigr\rangle
$$

を得る。
ここで $z_\theta$ に関して条件付き期待値を取ると、

$$
\mathbb{E}\Bigl[\|q(z_\theta)-z'_\xi\|_2^2 \mid z_\theta\Bigr]=\|q(z_\theta)-\mu(z_\theta)\|_2^2+\mathbb{E}\Bigl[\|\mu(z_\theta)-z'_\xi\|_2^2 \mid z_\theta\Bigr]+2\mathbb{E}\Bigl[\Bigl\langle q(z_\theta)-\mu(z_\theta),\mu(z_\theta)-z'_\xi\Bigr\rangle\mid z_\theta\Bigr]
$$

となる。

最後の交差項は 0 になる。実際、$q(z_\theta)-\mu(z_\theta)$ は $z_\theta$ の関数だから条件付き期待値の外に出せて、

$$
\mathbb{E}\Bigl[\Bigl\langle q(z_\theta)-\mu(z_\theta),,\mu(z_\theta)-z'_\xi\Bigr\rangle\mid z_\theta\Bigr]=\Bigl\langle q(z_\theta)-\mu(z_\theta),\mathbb{E}\bigl[\mu(z_\theta)-z'_\xi \mid z_\theta\bigr]\Bigr\rangle
$$

である。一方、 $(\ast)$ より、

$$
\mathbb{E}\bigl[\mu(z_\theta)-z'_\xi \mid z_\theta\bigr]=\mu(z_\theta)-\mathbb{E}[z'_\xi \mid z_\theta]=0
$$

だから、交差項は消える。したがって

$$
\mathbb{E}\Bigl[\|q(z_\theta)-z'_\xi\|_2^2 \mid z_\theta\Bigr]=\|q(z_\theta)-\mu(z_\theta)\|_2^2+\mathbb{E}\Bigl[\|\mu(z_\theta)-z'_\xi\|_2^2 \mid z_\theta\Bigr]
$$

を得る。
ここで右辺第2項は $q$ に依存しない。したがって、この量を最小にするには第1項を 0 にすればよい。ゆえに最適 predictor $q^\star$ は

$$
q^\star(z_\theta)=\mu(z_\theta)=\mathbb{E}[z'_\xi \mid z_\theta]\tag{4}
$$

で与えられる。
これは、$z_\theta$ が与えられたときに predictor が返すべき最良の予測は、$z'_\xi$ の条件付き期待値である、という意味である。

次に、この最適 predictor を損失に代入すると、何が残るかを見る。
ここで使うのは、二乗誤差の標準的な分解である。任意の確率変数 $X,Y$ と、$X$ から予測を与える関数 $a$ に対して、

$$
\underbrace{\mathbb{E}\Bigl[\|Y-a(X)\|_2^2 \mid X\Bigr]}_{\text{条件付き予測誤差}} = \underbrace{\mathbb{E}\Bigl[\|Y-\mathbb{E}[Y\mid X]\|_2^2 \mid X\Bigr]}_{\text{条件付き分散}} + \underbrace{\|\mathbb{E}[Y\mid X]-a(X)\|_2^2}_{\text{最適予測子からのずれ}}
$$

が成り立つ[^5]。ここで $a(X)$ は、予測関数 $a$ が入力 $X$ に対して返す予測値である。したがって、この式は「予測誤差は、$X$ を知っていても残るばらつき」と「予測関数 $a$ が最適予測子 $\mathbb{E}[Y\mid X]$ からどれだけずれているか」に分解できることを表している。

ここで

* $X=z_\theta$
* $Y=z'_\xi$
* $a(X)=q(z_\theta)$

と置けば、

$$
\mathbb{E}\Bigl[\|q(z_\theta)-z'_\xi\|_2^2 \mid z_\theta\Bigr]=\mathbb{E}\Bigl[\|z'_\xi-\mathbb{E}[z'_\xi\mid z_\theta]\|_2^2 \mid z_\theta\Bigr]+\|q(z_\theta)-\mathbb{E}[z'_\xi\mid z_\theta]\|_2^2
$$

となる。
右辺第2項は、predictor $q$ が最適 predictor からどれだけずれているかを表している。
したがって、 $q=q^\star$ のときこの項は 0 になり、最小達成値は

$$
\mathbb{E}\Bigl[\|z'_\xi-\mathbb{E}[z'_\xi\mid z_\theta]\|_2^2 \mid z_\theta\Bigr]
$$

だけになる。
さらに $q=q^\star$ のとき、全体で期待値を取ると

$$
\mathbb{E}\Bigl[\|q^\star(z_\theta)-z'_\xi\|_2^2\Bigr]=\mathbb{E}\Bigl[\|z'_\xi-\mathbb{E}[z'_\xi\mid z_\theta]\|_2^2\Bigr]
$$

を得る。
左辺は「最適 predictor を使ったときにもなお残る予測誤差」であり、右辺は「$z_\theta$ を知っていてもなお残る $z'_\xi$ のばらつき」である。

ここでノルム二乗を成分ごとに展開すると、

$$
\mathbb{E}\Bigl[\|z'_\xi-\mathbb{E}[z'_\xi\mid z_\theta]\|_2^2\Bigr]=\mathbb{E}\Bigl[\sum_i \mathrm{Var}(z'_{\xi,i}\mid z_\theta)\Bigr]
$$

となる。
ここで $z'_{\xi,i}$ は target 射影 $z'_\xi$ の第 $i$ 成分である。
したがって、**最適 predictor を仮定すると、BYOL の predictor 誤差は target 表現の条件付き分散そのものになる**。

この関係を用いると、$\theta$ に関する更新方向は

$$
\nabla_\theta\mathbb{E}\Bigl[\sum_i\mathrm{Var}(z'_{\xi,i}\mid z_\theta)\Bigr]\tag{5}
$$

と結びつけて読むことができる。

この式の意味は明確である。
BYOL は、単に target に近いベクトルを出力せよと要求しているのではない。そうではなく、**online 表現 $z_\theta$ を見れば、target 表現 $z'_\xi$ がなるべく予測しやすくなるようにせよ** と要求しているのである。

直感的には、$z_\theta$ が十分に情報を保持していれば、同じ $z_\theta$ に対応する $z'_\xi$ はあまり散らばらない。このとき、$z_\theta$ に条件づけた $z'_\xi$ のばらつき、すなわち条件付き分散は小さい。逆に、$z_\theta$ が情報を捨てすぎていれば、同じ $z_\theta$ に対して多様な $z'_\xi$ が対応してしまう。すると predictor は平均的な値しか返せず、条件付き分散は大きいまま残る。

したがって、この見方に立つと、BYOL は

$$
\begin{aligned}
&\textbf{単に target に一致する表現を作る}\\
&\text{のではなく、}\\
&\textbf{target の変動を説明できる表現を online 側に作る}
\end{aligned}
$$

方向に学習していると解釈できる。

ここで重要なのは、説明すべき対象が外部から与えられたラベルではないという点である。
BYOL が説明しようとしているのは、 **同じ入力の別 view に対して target network が与える表現** であり、その target 自体も EMA によってゆっくり変化していく。言い換えれば、BYOL は **自分自身の少し過去の表現が持っていた構造を、現在の online 表現でなるべく予測可能にする** ように学習しているのである。

この観点から見ると、崩壊表現が望ましくない理由も見えてくる。もしすべての入力がほぼ同じ表現に押しつぶされてしまえば、$z_\theta$ を見ても $z'_\xi$ の違いを説明できなくなる。すると、$z_\theta$ に条件づけたときの $z'_\xi$ のばらつきは大きいままであり、良い predictor も作れない。その意味で BYOL は、単なる自己一致ではなく、 **予測可能性を保つだけの情報を表現に残すよう圧力をかけている** と読むことができる。

もっとも、この議論には注意も必要である。論文では、ここで述べた崩壊回避のメカニズムを完全な定理として証明しているわけではない。主張の骨格は、predictor が十分に最適に近いとき、BYOL の更新が条件付き分散を減らす方向として読める、という点にある。そこから、定数表現のような極端に情報を失った表現は target の変動をうまく説明できず、したがって望ましい平衡点にはなりにくい、という理論的直感が導かれる。したがって、BYOL がなぜ崩壊しないのかは、厳密に解き切られた定理として提示されているというより、非対称な更新ダイナミクスと条件付き分散の観点から整合的に説明されている と理解するのが誠実である。

[^5]: 参考資料: [CMU Lecture Notes: Optimal Prediction](https://www.stat.cmu.edu/~larry/=stat401/lecture-01.pdf)

### 4.4 target network は何をしているのか

ここまでで、predictor が target を予測する役割を担っていることは見えてきた。では、target network 自体は何をしているのだろうか。

論文の重要な指摘は、**target が速く変化しすぎると、predictor がほぼ最適に保たれているという前提が崩れる**という点にある。もし target が各ステップで大きく動いてしまえば、online 側は絶えず変化する相手を追い続けることになり、predictor はその時々の target を安定に近似できない。すると、前節で見た **条件付き分散にもとづく解釈** も支えを失う。

というのも、あの議論は本質的に、 **「現在の online 表現が与えられたとき、predictor は target 側の変動を十分うまく説明できる」** という状況を仮定しているからである。記号で書けば、target 側の表現を $Z_\xi$ 、online 側の表現を $Y_\theta$ 、predictor 出力を $q_\theta(Y_\theta)$ としたとき、議論の背後には

$$
q_\theta(Y_\theta) \approx \mathbb{E}[Z_\xi \mid Y_\theta]
$$

のような「条件付き期待値への近似」という見方がある。ところが $Z_\xi$ 自体が急速に動いてしまえば、この近似はその都度崩れやすくなり、結果として

$$
\operatorname{Var}(Z_\xi \mid Y_\theta)
$$

を小さくする方向に学習が進む、という解釈も弱くなる。

一方、target を EMA によってゆっくり更新すれば、predictor はその変化に追従しやすくなり、near-optimality な状態を保ちやすい。論文はこの意味で、 **moving average target network の主たる役割は、学習を通じて predictor の near-optimality を保ちやすくすることにある** と述べている。

これは BYOL の理解において本質的な点である。EMA target の役割は、単に「過去の自分を教師にする」ことではない。より重要なのは、 **予測問題そのものを安定に保つこと** にある。

BYOL の学習は、概念的には次の循環として捉えられる。

1. online network が target representation を予測する
2. predictor がその予測問題を解く
3. target は EMA によりゆっくり変化するため、predictor は追従可能である
4. その結果、online 側には target の変動を説明できる表現を保つ圧力がかかる

つまり target network は、教師信号を与えるだけでなく、 **その教師信号の時間変化を制御することで、条件付き分散の議論が意味を持つ学習を維持している** のである。

### 4.5 この節の結論

以上をまとめると、BYOL が崩壊しにくい理由について、論文は次のような理解を与えている。

1. **BYOL は $(\theta, \xi)$ の単純な joint minimization ではない。**
   target 側は勾配で最適化されず、EMA で更新される。したがって、「崩壊解が損失最小だから学習もそこへ向かう」という素朴な議論は、そのままでは当てはまらない。

2. **predictor が十分よく最適化されているなら、BYOL は条件付き分散を減らす方向に解釈できる。**
   この見方では、online 表現は target の変動を説明する情報を保持する方向に押されるため、定数表現は安定な解になりにくい。

3. **EMA target の主たる役割は、predictor が near-optimality でいられるように予測問題を安定化することにある。**
   predictor / stop-gradient / EMA / target は独立な小技ではなく、学習を成立させるための相互補完的な設計要素である。

BYOL の面白さは、「なぜ負例なしで学習できるのか」という問いに対して、 **負例の代わりに何を入れたか** ではなく、 **どのような非対称な動的系を設計したか** という観点から答えている点にある。

---

## 5. SimCLRと何が違うのか

### 5.1 SimCLRの原理：比較による表現学習

SimCLR の基本発想は、表現学習を **識別問題** として定式化することにある。同じ画像から得た 2 つの view を正例として近づけ、異なる画像に由来する view を負例として遠ざけることで、表現空間に識別的な構造を与える。典型的には、その学習信号は次のような contrastive loss で表される。

$$
\mathcal{L}_{\mathrm{SimCLR}}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}
$$

ここで $z_i, z_j$ は同一画像から得た 2 つの view の表現、 $\operatorname{sim}(\cdot,\cdot)$ は cosine similarity 、 $\tau$ は温度パラメータである。この式の本質は明快で、分子は「対応する相手を選ぶ」項、分母は「他の候補と区別する」項になっている。

したがって SimCLR では、表現が情報を保持する理由はかなりはっきりしている。もしすべての入力が同じ表現に潰れてしまえば、正例と負例を区別できず、識別課題そのものが成立しない。つまり SimCLR における anti-collapse の原理は、 **比較構造そのものの中に埋め込まれている** 。

その反面、この設計は比較対象の作り方に強く依存する。負例が十分に豊かでなければ識別課題は弱くなり、augmentation が不適切であればショートカットで課題が解けてしまう。SimCLR において大きな batch size や慎重な augmentation 設計が重要になるのは、このためである。

要するに、SimCLR の表現学習原理は「同じものを近づけ、違うものを遠ざける」という比較ベースの構図にある。

### 5.2 BYOLの原理：安定化された自己予測

これに対して BYOL は、表現学習を識別問題としてではなく、 **自己予測問題** として構成する。
online network は、一方の view から得た表現を用いて、他方の view に対する target network の表現を予測する。抽象化すると、その中心は次のような目的にある。

$$
\mathcal{L}_{\mathrm{BYOL}}=\left\|q_\theta(z_\theta)-z'_\xi\right\|_2^2
$$

ここで $z_\theta$ は online encoder + projector による表現、 $z'_\xi$ は target encoder + projector による表現、 $q_\theta$ は predictor である。重要なのは、BYOL が「正しい相手を他と見分ける」問題を解いているのではなく、 **自分の少し遅れて更新される表現を予測する** 問題を解いているという点である。

この違いは本質的である。SimCLR では、表現に情報を保持させる圧力は「他サンプルと区別しなければならない」という外在的な比較から生じる。これに対して BYOL では、その圧力は「target が持っている変動を予測できなければならない」という内在的な予測制約から生じる。前節で見た条件付き期待値・条件付き分散の議論は、まさにこの自己予測の力学を説明するためのものだった。

もちろん、この予測問題はそのままでは不安定である。BYOL はそれを成立させるために、 **predictor / stop-gradient / EMA target** という非対称な安定化機構を導入している。したがって BYOL を単に「負例を除いた SimCLR」とみなすのは正確ではない。両者の違いは負例の有無だけではなく、 **学習信号の源泉が比較か予測か** という点にある。

### 5.3 batch size 依存の違いは何を意味しているのか

この原理差は、batch size に対する感度の違いとしても現れる。
SimCLR では、学習信号の一部が負例集合の豊かさに依存しているため、batch size が小さくなると一度に比較できる候補が減り、識別課題そのものが弱くなる。したがって、小バッチで性能が落ちやすいのは偶然ではなく、比較ベースの学習原理の自然な帰結である。

これに対して BYOL では、学習信号の中心は負例との比較ではない。必要なのは、多数の他サンプルを同時に並べることではなく、予測すべき target と、それを安定に追従できる力学である。そのため、BYOL の batch size 依存は SimCLR とは性質が異なる。実際、論文でも SimCLR は小バッチで大きく悪化する一方、BYOL はより広い範囲で安定していることが報告されている。

もちろん、これは BYOL が batch size に全く依存しないという意味ではない。非常に小さい batch では batch normalization などを通じた不安定性が生じうる。重要なのは、**その劣化が「負例不足」では説明されない** ことである。ここに、両者の学習原理の差が端的に現れている。

### 5.4 augmentation 依存の違いは何を意味しているのか

augmentation に対する依存の仕方も、同じ対比で理解できる。
SimCLR では augmentation は単なるデータ拡張ではなく、 **識別課題をどう作るか** を決める核心的な要素である。もし同じ画像の 2 view が色ヒストグラムのような単純な統計だけで容易に対応づけられてしまうなら、モデルはそのショートカットで目的を達成できる。その結果、表現はより本質的な情報を保持する必要がなくなる。SimCLR において color distortion が重要になるのはこのためである。

一方、BYOL では目標は「他と見分けること」ではなく、「target 表現を予測すること」にある。したがって、単純な統計量だけに依存して課題を解き切る構図になりにくい。online 側は、target が持つ変動を予測するために、それを支える情報を表現内に残す方向へ圧力を受ける。論文で、BYOL の方が augmentation の選択に対して頑健であると報告されているのは、この予測ベースの構造と整合的である。

ここでも重要なのは、同じ augmentation を使っていても、その役割が異なるという点である。SimCLR では augmentation は識別問題の難しさを規定する設計変数だが、BYOL では予測すべき target を生成するための view 生成手段である。この違いが、augmentation 依存の差として観測される。

### 5.5 BYOLはSimCLRの変種ではなく、別原理である

以上を踏まえると、BYOL は SimCLR の単なる改良版ではない。より正確には、 **崩壊を防ぐ原理を比較から予測へと置き換えた手法** である。

SimCLR を支えるのは、他サンプルとの比較による識別制約である。
BYOL を支えるのは、非対称に安定化された自己予測である。

前者では、

$$
\textbf{他サンプルと区別できなければならない}
$$

という制約が表現を支える。
後者では、

$$
\textbf{target が持つ変動を予測できなければならない}
$$

という制約が表現を支える。

この違いを見落とすと、BYOL は「負例を消した SimCLR」に見えてしまう。しかし実際には、学習信号の源泉そのものが異なる。そのため、batch size や augmentation に対する依存の仕方まで変わってくるのである。

### 5.6 この節の結論

この章の結論は明確である。
SimCLR は **識別** によって崩壊を防ぎ、BYOL は **安定化された自己予測** によって崩壊を防ぐ。したがって両者の違いは、単なる実装上の差分ではない。batch size や augmentation に対する振る舞いの違いも含めて、それは **学習原理の差** として理解すべきである。

BYOL の意義は、単に負例を不要にしたことにあるのではない。むしろ、 **自己教師あり学習の核を識別以外の原理でも構成できる** ことを示した点にある。

---

## 6. アブレーションで見るBYOLの本質

BYOL の面白さは、最終精度の高さだけにあるのではない。むしろ重要なのは、 **どの要素を外すと壊れ、どの要素を残すと成立するのかが、アブレーションを通してかなり明瞭に見える** ことである。論文は 5 節で、batch size、augmentation、target decay、predictor の有無、contrastive 成分との比較などを系統的に調べており、BYOL の成立条件を丁寧に可視化している。

ここで重要なのは、表や数値を個別に追うことではない。見るべきなのは、 **それぞれのアブレーションが BYOL の設計原理をどう支持しているか** である。この章では、その観点から結果を読み直す。

### 6.1 batch size と augmentation に対する頑健性は、学習信号の違いを表している

まず目につくのは、BYOL が batch size と augmentation の両方に対して、SimCLR より頑健に振る舞うことである。論文では、batch size を変えたときの性能低下と、augmentation を弱めたときの性能低下を比較しており、どちらの場合も SimCLR は大きく崩れる一方、BYOL は相対的に安定していることが示されている。

![](https://storage.googleapis.com/zenn-user-upload/ceba0510508c-20260410.png)
*Fig3. ImageNet 上で線形評価を行った場合の、学習 300 エポック時点における BYOL および著者らが再現した SimCLR の top-1 精度（%）。
*

この二つの結果は別々の話ではない。どちらも、 **BYOL が負例比較を学習信号の中心に置いていない** ことの帰結として理解できる。SimCLR では、学習信号の質が「どれだけ十分な識別問題を作れるか」に強く依存する。したがって batch size が小さくなると、一度に比較できる負例が減り、識別課題そのものが弱くなる。また augmentation が不適切だと、モデルは色ヒストグラムのような表層的な統計だけで正例対応を見抜けてしまい、表現はより本質的な情報を保持する必要がなくなる。つまり SimCLR において batch size と augmentation は、どちらも **識別問題の強さを規定する設計変数** として働いている。

これに対して BYOL は、他サンプルとの比較ではなく、別 view に対する target 表現の予測によって学習する。そのため、大量の負例を同時に並べること自体は本質ではなく、また表面的な手掛かりだけで学習が済んでしまう構図にもなりにくい。online 側は、target が保持している変動を予測するために、それを支える情報を表現内に残しておく必要がある。したがって BYOL の batch size 頑健性と augmentation 頑健性は、どちらも偶然の性質ではなく、 **学習信号の源泉が contrastive な比較とは異なる** ことの表れだと読める。

もちろん、これは BYOL が batch size や augmentation に全く依存しないという意味ではない。論文でも、ごく小さい batch では batch normalization の影響による性能低下が指摘されているし、augmentation を極端に弱めれば性能は落ちる。重要なのは、 **その崩れ方の理由が SimCLR と異なる** ことである。SimCLR は主として識別問題の弱体化によって崩れ、BYOL は主として正規化や最適化、あるいは予測対象の情報量低下によって崩れる。ここに、両者の原理差がはっきり現れている。

### 6.2 EMA target・predictor・負例項のアブレーションが示すもの

batch size や augmentation の結果が、BYOL と SimCLR の **学習信号の違い** を示していたのに対し、Table 1 のアブレーションは、BYOL の内部で何が本質的に効いているのかをより直接に示している。ここで論文は、target network の更新速度だけでなく、predictor の有無、target network の有無、さらに負例項の有無まで含めて、BYOL と SimCLR を同じ枠組みで比較している。

その際に導入されるのが $\beta$ である。論文は SimCLR と BYOL を、拡張された InfoNCE 型の目的関数の中で統一的に書き直しており、 $\beta \in [0,1]$ はその中で **負例項にかかる重み係数** を表す。直感的には、 $\beta = 1$ なら負例項を含む contrastive learning 側の設定、 $\beta = 0$ なら負例項を取り除いた bootstrap / 自己予測側の設定に対応する。したがって表 5b は、predictor や target network の有無だけでなく、 **学習信号の中に負例比較を残すかどうか** まで含めて比較していることになる。

![](https://storage.googleapis.com/zenn-user-upload/a4752b24914e-20260410.png)
*Table1. ImageNet 上で線形評価を行った場合の、学習 300 エポック時点における top-1 精度（%）のアブレーション結果*

まず Table 1.(a) は、target network が存在することだけでなく、 **どの速さで更新されるか** が本質的であることを示している。 $\tau = 0$ 、すなわち target を毎ステップ online に即時同期すると、top-1 精度は 0.3% まで落ち、学習はほぼ崩壊する。逆に $\tau = 1$ 、すなわち target を全く更新しない定数ネットワークにすると、学習自体は安定するものの、精度は 18.8% にとどまる。これに対して、online network の移動平均として target を 0.9〜0.999 程度で更新すると、68.4〜72.5% に達する。つまり、target を速く動かしすぎると online 側が追う相手が変わりすぎ、逆に固定しすぎるとブートストラップによる反復的改善が止まる。EMA target は、単に「過去の自分を使う」ための仕掛けではなく、 **予測問題を適切な難しさと安定性の範囲に保つ速度制御** として働いているのである。

次に表 5b は、predictor / target network / 負例項の三者がどのように役割分担しているかを示している。ここで最も重要なのは、 **負例なし $(\beta = 0)$ で高性能に動作するのは、predictor と target network の両方を備えた BYOL だけ** だという点である。実際、BYOL は predictor あり・target network あり・ $\beta = 0$ で 72.5% を達成する。しかし、SimCLR では $\beta = 0$ の条件下で predictor を外したり、target network を外したりすると、0.3%、0.2%、0.1% とほぼ完全に崩壊する。これは、負例を消すだけでは学習は成立せず、 **online 側に非対称性を与える predictor と、安定した予測対象を与える target network の両方が必要** であることを示している。

この結果から、BYOL の成功を「EMA が効いている」「predictor が効いている」といった単一要因で説明することはできない。より正確には、

* predictor は online 側にのみ片側の変換自由度を与え、構造的な非対称性を導入する
* target network は、online より遅れて変化する安定な予測対象を供給する
* stop-gradient は、回帰誤差に対する直接の更新を online 側に限定し、target 側を追われる参照先として保つ

という役割分担があって初めて、負例なしの自己予測が崩壊せずに成立する。したがって、BYOL の anti-collapse 機構は単独のトリックではなく、 **非対称な自己予測系としての相互作用** によって成立していると理解すべきである。

### 6.3 この章の結論

アブレーションを通して見えてくる BYOL の本質は、かなり明快である。

1. **batch size に対する頑健性** は、BYOL が負例比較を学習信号の中心に置いていないことを示している。
2. **augmentation に対する頑健性** は、BYOL の表現が識別問題ではなく、target 予測によって駆動されていることを示している。
3. **target decay に適切な値が必要であること** は、EMA target が予測問題を安定化するための速度制御として機能していることを示している。
4. **predictor と target network の両方が必要であること** は、BYOL が単一の技巧ではなく、非対称な自己予測系として成立していることを示している。

したがって、BYOL の成功を最終精度の高さだけで語るのは不十分である。本当に重要なのは、アブレーションを通して

$$
\textbf{何を外すと崩壊し、何を残すと予測が成立するのか}
$$

がかなり鮮明に見える点にある。BYOL が示したのは、自己教師あり学習において、強い表現学習は必ずしも比較に依存しなくてもよいということである。すなわち、 **適切に安定化された非対称な自己予測によっても、高品質な表現は十分に獲得できる** 。BYOL の意義は、この可能性を理論的直観と実験の両面から具体的に示したことにある。

---

## 7. まとめ

BYOL はしばしば「負例を使わない自己教師あり学習法」として紹介される。もちろんそれは事実である。しかし、その説明だけではこの論文の本質には届かない。BYOL の重要性は、単に負例を取り除いたことではなく、 **それでも崩壊しない学習系をどのように設計したか** にある。論文が示したのは、自己教師あり学習の成立条件が「負例をどう集めるか」だけに尽きない、ということであった。

本記事で見てきたように、BYOL は online network と target network の非対称な相互作用の上に成り立っている。online 側は損失に対して勾配で更新される一方、target 側はその指数移動平均として更新される。したがって、BYOL を単純な joint minimization として捉えるのは正確ではない。むしろ重要なのは、 **誰が損失に直接適応し、誰が遅れて追随するのか** という更新構造である。崩壊の有無は、静的な目的関数の形だけでなく、この非対称な更新則に強く依存している。

さらに論文は、predictor が最適に近いと仮定したとき、BYOL の更新を target 表現に対する **条件付き分散を減らす方向** として解釈できることを示す。これにより、online 表現は単に target に一致すればよいのではなく、target が持つ変動を説明できるような情報を保持する方向へ押される、と理解できる。したがって、定数表現のような崩壊解は望ましい固定点ではなく、不安定だと考えられる。もっとも、これは完全な定理として与えられているわけではなく、最適 predictor の仮定のもとで得られる理論的直観である点には注意が必要である。

実験面でも、この設計思想はかなり一貫して支持されている。batch size に対する相対的な頑健性は、学習信号の中心が負例比較ではないことを示している。augmentation に対する頑健性は、表層的な統計的手掛かりだけではなく、target 予測によって表現が駆動されていることを示している。さらに、target decay の実験は「ゆっくり動く教師」が必要であることを示し、predictor と target network に関するアブレーションは、BYOL が単一の技巧ではなく、 **predictor / stop-gradient / EMA target の組合せ** として成立していることを明らかにしている。

この観点から見ると、BYOL は SimCLR の単なる改良版ではない。SimCLR が「識別」によって崩壊を防いでいたのに対し、BYOL は「安定化された自己予測」によって崩壊を防いだ。言い換えれば、BYOL は自己教師あり学習の中心原理を、 **比較** から **予測と安定化** へと拡張した手法である。だからこそこの論文は、その後の non-contrastive learning や self-distillation 系の流れを理解するうえでの重要な分岐点になっている。

もちろん、BYOL でも 2 つの異なる view を作るために augmentation は依然として重要である。ただし contrastive learning と異なり、学習の安定性や崩壊回避を入力側の設計だけに委ねているわけではない。BYOL では、EMA による target network、stop-gradient、predictor からなるモデル側の非対称な構造そのものが学習信号を整えており、そのぶん augmentation や大規模な負例設計への依存は相対的に弱くなっている。

要するに、BYOL の本質は「負例がなくても学習できた」という事実ではなく、 **非対称性と安定化によって自己予測を成立させ、表現学習の新しい成立条件を示したことにある** 。