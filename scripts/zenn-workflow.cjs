#!/usr/bin/env node
'use strict';

// Local draft/figure preparation only. Publication is always the author's job.
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const projectRoot = path.resolve(__dirname, '..');
const cliPath = path.join(projectRoot, 'node_modules', 'zenn-cli', 'dist', 'server', 'zenn.js');
const maxImageBytes = 3_000_000;
const chapters = ['はじめに', '問題設定と前提', '手法と導出', '理論的な解釈', '実験・限界・思想', 'さいごに'];

function fail(message) { throw new Error(message); }

function help() {
  console.log(`Zenn の未公開記事と図をローカルで準備します。

  node scripts/zenn-workflow.cjs new --slug <slug> --title <タイトル> [--topics <topic1,topic2>]
  node scripts/zenn-workflow.cjs figure --slug <slug> --source <画像パス> --name <figure.png>
  node scripts/zenn-workflow.cjs help

slug: 半角小文字・数字・ハイフン・アンダースコアの 12〜50 文字。
topics: 最大 5 個。記事は必ず published: false で生成します。
figure: PNG/JPEG/GIF/WebP、3,000,000 bytes 以下。既存ファイルは上書きしません。
図は images/zenn/<slug>/ にコピーし、Markdown を表示します。アップロードは行いません。
Git の add / commit / push、公開操作、依存パッケージの自動取得は行いません。
--root <ディレクトリ> で保存先を変更できます（articles/ が必要）。`);
}

function parseOptions(command, args) {
  const allowed = new Set(command === 'new'
    ? ['slug', 'title', 'topics', 'root'] : ['slug', 'source', 'name', 'root']);
  const result = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].startsWith('--') ? args[i].slice(2) : '';
    if (!allowed.has(key)) fail(`未対応の引数です: ${args[i]}`);
    if (Object.hasOwn(result, key)) fail(`引数が重複しています: --${key}`);
    if (i + 1 >= args.length || args[i + 1].startsWith('--')) fail(`--${key} の値が必要です。`);
    result[key] = args[i + 1];
  }
  for (const key of command === 'new' ? ['slug', 'title'] : ['slug', 'source', 'name']) {
    if (!result[key]?.trim()) fail(`--${key} の値が必要です。`);
  }
  if (!/^[a-z0-9_-]{12,50}$/.test(result.slug)) fail('slug は半角小文字・数字・ハイフン・アンダースコアの 12〜50 文字にしてください。');
  return result;
}

// Reject symlink/junction components, so an existing directory cannot redirect writes.
function localDirectory(root, parts, create = false) {
  let current = root;
  for (const part of parts) {
    current = path.join(current, part);
    try {
      const info = fs.lstatSync(current);
      if (info.isSymbolicLink() || !info.isDirectory()) fail(`通常のディレクトリが必要です: ${current}`);
    } catch (error) {
      if (error.code !== 'ENOENT' || !create) throw error;
      fs.mkdirSync(current);
    }
  }
  return current;
}

function reportIgnored(root, relativePath) {
  const result = spawnSync('git', ['check-ignore', '--', relativePath], {
    cwd: root, encoding: 'utf8', windowsHide: true, shell: false,
  });
  if (result.status === 0) {
    console.log(`注意: ${relativePath} は Git の除外対象です。Git 連携ではこの状態のまま配信されません。`);
  } else if (result.error || result.status !== 1) {
    console.log('Git の除外状態は確認できませんでした（Git 未導入、または Git 管理外など）。');
  }
}

function newArticle(root, options) {
  const articlePath = path.join(localDirectory(root, ['articles']), `${options.slug}.md`);
  if (fs.lstatSync(articlePath, { throwIfNoEntry: false })) fail(`記事は既に存在します。上書きしません: ${articlePath}`);
  if (/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/.test(options.title)) fail('タイトルに改行や制御文字は使用できません。');
  const topics = options.topics === undefined ? [] : options.topics.split(',').map(topic => topic.trim());
  if (topics.length > 5 || topics.some(topic => !topic || /[\u0000-\u001f\u007f-\u009f]/.test(topic))) {
    fail('topics は空要素を含まない最大 5 個のカンマ区切りにしてください。');
  }
  if (!fs.existsSync(cliPath)) fail('インストール済みの zenn-cli が見つかりません。プロジェクトの依存関係を確認してください。');
  const result = spawnSync(process.execPath, [cliPath, 'new:article', '--slug', options.slug,
    '--title', options.title.replace(/\\/g, '\\\\'), '--published', 'false', '--machine-readable'], {
    cwd: root, encoding: 'utf8', windowsHide: true, shell: false,
  });
  // zenn-cli may report a failure with exit code 0, so inspect both its output and the file.
  const expectedOutput = `articles/${options.slug}.md`;
  const hasExpectedOutput = result.stdout?.split(/\r?\n/).some(line => line.trim() === expectedOutput);
  if (result.error || result.status !== 0 || !hasExpectedOutput || !fs.existsSync(articlePath)) {
    fail(`zenn-cli による記事作成を確認できませんでした。\n${result.error?.message || result.stderr || result.stdout || ''}`.trim());
  }
  const info = fs.lstatSync(articlePath);
  if (!info.isFile() || info.isSymbolicLink()) fail('生成先が通常のファイルではありません。');
  const generated = fs.readFileSync(articlePath, 'utf8');
  if (!/^---\r?\n[\s\S]*\r?\n---\r?\n?$/.test(generated) || !/^published: false\r?$/m.test(generated)
    || !/^title: /m.test(generated) || !/^topics: \[\]\r?$/m.test(generated)) {
    fail('想定した未公開フロントマターではありません。生成ファイルを確認してください。');
  }
  const frontmatter = generated.replace(/^title: .*$/m, () => `title: ${JSON.stringify(options.title)}`)
    .replace(/^topics: \[\]\r?$/m, () => `topics: ${JSON.stringify(topics)}`);
  const outline = chapters.map((title, index) => `## ${index + 1}. ${title}\n`).join('\n');
  fs.writeFileSync(articlePath, `${frontmatter.trimEnd()}\n\n${outline}`, 'utf8');
  console.log(`未公開記事を作成しました: ${articlePath}\n2〜5 章の見出しは仮置きです。各章の節は最大 3 つを目安に編集してください。`);
  reportIgnored(root, expectedOutput);
}

function imageType(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'png';
  if (bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'jpeg';
  if (['GIF87a', 'GIF89a'].includes(bytes.toString('ascii', 0, 6))) return 'gif';
  if (bytes.length >= 12 && bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') return 'webp';
  return null;
}

function prepareFigure(root, options) {
  if (!/^[a-z0-9][a-z0-9_-]*\.(png|jpe?g|gif|webp)$/.test(options.name)
    || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])\./.test(options.name)) {
    fail('name は半角小文字・数字・ハイフン・アンダースコアと画像拡張子で指定してください。パスや Windows 予約名は使用できません。');
  }
  const sourcePath = path.resolve(options.source);
  const info = fs.statSync(sourcePath);
  if (!info.isFile() || info.size > maxImageBytes || info.size === 0) fail('画像は空でない通常のファイルで、3,000,000 bytes 以下にしてください。');
  const bytes = fs.readFileSync(sourcePath);
  const extension = path.extname(options.name).slice(1).replace(/^jpg$/, 'jpeg');
  if (bytes.length > maxImageBytes || imageType(bytes) !== extension) fail('画像の内容と保存先の拡張子が一致しないか、対応形式ではありません。');
  const directory = localDirectory(root, ['images', 'zenn', options.slug], true);
  const destination = path.join(directory, options.name);
  fs.writeFileSync(destination, bytes, { flag: 'wx' });
  const relativePath = `images/zenn/${options.slug}/${options.name}`;
  console.log(`図をローカルに配置しました（未アップロード）: ${destination}\n\n![図の説明を記入](/${relativePath})\n`);
  reportIgnored(root, relativePath);
}

try {
  const [command = 'help', ...args] = process.argv.slice(2);
  if (['help', '--help', '-h'].includes(command)) {
    help();
  } else {
    if (!['new', 'figure'].includes(command)) fail(`未対応のコマンドです: ${command}（help を参照）`);
    const options = parseOptions(command, args);
    const root = fs.realpathSync(path.resolve(options.root || projectRoot));
    localDirectory(root, ['articles']);
    (command === 'new' ? newArticle : prepareFigure)(root, options);
  }
} catch (error) {
  console.error(`エラー: ${error.code === 'EEXIST' ? '保存先は既に存在します。上書きしません。' : error.message}`);
  process.exitCode = 1;
}
