---
name: zenn-article-workflow
description: この zenn-docs でZenn記事の下書き生成、図のローカル配置、Markdown・KaTeXの表示確認、GitHub画像連携の準備を行う。記事作成やfigure追加をコマンドで進めたいときに使い、公開状態の変更は作者に残す。
---

# Zenn の下書きと図を扱う

リポジトリルートで実行する。`AGENTS.md` の公開担当と既存ファイルの扱いを守る。執筆内容の判断には [zenn-paper-article](../zenn-paper-article/SKILL.md)、記法には [zenn-markdown.md](references/zenn-markdown.md) を使う。

## 下書きを生成する

まず `git status --short` で進行中の作業を把握し、既存記事の更新か新規作成かを判断する。既存記事は同じファイルを編集する。

```powershell
node scripts/zenn-workflow.cjs new --slug model-soup-notes --title "Model Soup の数理と思想" --topics "機械学習,深層学習"
```

補助スクリプトはインストール済みの Zenn CLI を呼び、`published: false` の記事と6章の骨組みを作る。第2〜5章の題名は題材に合わせて変更する。slug は半角小文字英数字・`-`・`_` の12〜50文字。既存ファイルは上書きしない。

公式CLIを直接使う場合は次の形になる。

```powershell
npx --no-install zenn new:article --slug model-soup-notes --title "Model Soup の数理と思想" --type tech --emoji "📘" --published false
```

ローカルにCLIがなければ、このリポジトリの依存設定を確認して `npm install` する。既存の `package.json` はローカルで使われているがGit管理対象外である。記事作成のために依存設定を一括変更しない。CLIのバージョン変更時は `--help` で利用するオプションを再確認する。

CLIが終了しても作成に成功したとは限らない。期待するファイルの存在、frontmatter、`published: false` を確認する。`published: true` や公開日時をエージェントが書き込む運用は設けない。

## 図を準備する

まず本文のどの主張・比較・仕組みを理解するために必要かを決め、使う図表を選ぶ。原論文や資料フォルダの図表をすべて掲載しない。表は画像にせずMarkdownで書き、抜粋した行・列、単位、評価条件、出典が分かるようにする。

原論文の図を示す場合はスクリーンショットを使う。作者が用意したスクリーンショットがあれば内容と原図を照合して優先し、なければ該当PDFページから取得する。元のページ・Figure番号・軸・凡例を確認し、必要な情報を切り落とさない。原図を新しく描き直したり、画像生成AIやMermaidで図を作ったりしない。

原図だけでは伝わりにくい点を補う可視化はPythonのmatplotlib / seabornで作り、コードを `src/` または `scripts/` の記事に対応する場所へ保存する。計算条件と再生成方法を残し、模式図と実測を区別する。図中の軸・凡例・動きと本文の説明を対応づける。動きの理解に必要な場合のみGIFを選ぶ。

静止図・GIFとも各ファイル3 MB以内にする。書き出したファイルのバイト数を確認し、超える場合は可読性を保ちながら画像サイズ、GIFのフレーム数・色数などを調整して再出力する。GIFは実際に再生し、動き・速度・ループを確認する。

```powershell
node scripts/zenn-workflow.cjs figure --slug model-soup-notes --source "images/model-soup-fig1.png" --name fig-01.png
```

`--source` には対象論文のスクリーンショット、またはPythonのmatplotlib / seabornで作った補足図の実際のパスを指定する。

補助スクリプトは `images/zenn/<slug>/<name>` にコピーし、形式・容量・上書きの有無を検査して Markdown を出力する。対応形式は PNG / JPG / JPEG / GIF / WebP。Zennの上限は1ファイル3 MBで、補助スクリプトは保守的に3,000,000バイト以下に制限する。

```markdown
![図の内容を説明する代替テキスト](/images/zenn/model-soup-notes/fig-01.png)
*Fig.1：何を比較した図か。軸・色の意味と、原論文の Figure 番号・出典。*
```

キャプションに加え、本文で「何を見るか → 何が観察できるか → 何を支持し、何までは言えないか」を説明する。原論文のスクリーンショットにはFigure番号と出典を付し、切り抜きや縮小を行った場合はその旨を示す。Pythonによる補足図は自作と明記し、計算条件を添える。図が未入手なら実在する画像URLを装わず、未完箇所として報告する。

**このコマンドはローカル配置であり、アップロードではない。** GitHub 連携では `/images/` の画像をGitへ追加し、Zennの連携先へ push して初めて同期される。相対URL `../images/...` は使わない。既存の `static.zenn.studio` などの画像URLはそのまま使える。

現在のCLIには `upload` コマンドや公開された公式CLI用画像upload APIを確認できていない。存在する前提でコマンドや認証手順を作らない。直接アップロードを希望された場合はZennの公式アップローダーを使う経路を調べ、確認できた機能だけで進める。

## 表示と差分を確認する

```powershell
npx --no-install zenn preview --host 127.0.0.1 --port 8000
```

起動ログでURLを確認し、対象記事をブラウザで開く。数式のエラー・横幅、図の読みやすさ、Markdown表、キャプション、折りたたみ、脚注、カード、章と節の表示を確認する。起動しただけで表示確認済みとは報告しない。CLIの `validate` は存在しないため、架空の検証コマンドを使わない。

```powershell
git diff --check
git diff -- articles/<slug>.md
git status --short
```

新規ファイルは通常の `git diff` には出ないので、本文も直接読む。数式の表示確認と数学的な正しさの確認は別々に行う。依頼されたコードの変更がある場合だけ該当する検証を追加する。

## 同期を依頼された場合

対象の記事と画像のパスを列挙して差分を確認する。`git add .` で作業素材・文献・別記事を巻き込まない。`paper/` と旧画像素材の無視設定を保ち、対象画像がGitで無視されていないことを `git check-ignore` で確認する。

新規記事は引き続き `published: false` を維持する。既存の公開記事が差分にある場合、pushによってその本文も更新されるので、依頼の対象に含まれるか確認する。連携先ブランチはZennの設定や既存運用から確かめ、現在のブランチが連携先だと決めつけない。

commit / push は依頼に含まれるときに行う。既に依頼されている操作を再承認待ちにしない。Gitの成功とZennの同期成功を区別し、確認できた段階だけを報告する。作者が自分で公開フラグを変更するまで、公開を代行しない。

## 確認した仕様

2026-09-28 に公式資料とローカルの `zenn-cli 0.5.2` を確認。仕様が異なる場合は公式資料と実際の `--help` を優先して手順を更新する。

- [Zenn CLI の使い方](https://zenn.dev/zenn/articles/zenn-cli-guide)
- [GitHub リポジトリの画像を使う](https://zenn.dev/zenn/articles/deploy-github-images)
- [Zenn の Markdown 記法](https://zenn.dev/zenn/articles/markdown-guide)
