# lecun1989-repro — C/C++ 移植版

*[English README is here → `README.md`](README.md)*

Andrej Karpathy による Yann LeCun et al. 1989
[Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
（誤差逆伝播法で訓練された、事実上最初の実用ニューラルネット）の再現実装を、
**依存ライブラリなしの標準 C++17** に移植したものです。

元の [`repro.py`](../repro.py) / [`prepro.py`](../prepro.py)（PyTorch）を
自己完結した C++ に移植しています。

- **PyTorch も BLAS も外部ライブラリも不要** — C++ 標準ライブラリのみ。
- **MSVC（Visual Studio 2022）, GCC, Clang** で、Windows / Linux / macOS
  上でそのままビルド・実行できます。
- 畳み込み、H2 層の疎結合（12 面中 8 面）、位置ごとの（共有しない）バイアス、
  各層の tanh 活性化、平均二乗誤差の損失、論文の重み初期化、素の SGD を、
  **順伝播も逆伝播もすべて手書き**で再実装しています。
- パラメータ数はちょうど **9760**（論文と一致）。

---

## 1. ビルド

### A. Visual Studio 2022（CMake — 最も簡単）

VS2022 は CMake プロジェクトを直接開けます。

1. Visual Studio 2022 を起動 → **ローカル フォルダーを開く** → この `cpp` フォルダを選択。
2. VS が `CMakeLists.txt` を自動検出します。構成を **`x64-Release`** に切り替えます。
3. **ビルド → すべてビルド**。
4. `lecun1989.exe` と `predict.exe` ができます。

### B. Visual Studio 2022（コマンドライン）

**x64 Native Tools Command Prompt for VS 2022** を開きます。

```bat
cl /O2 /EHsc /std:c++17 lecun1989.cpp
cl /O2 /EHsc /std:c++17 predict.cpp stb_impl.cpp
```

### C. CMake（任意のコンパイラ）— `lecun1989` と `predict` の両方をビルド

```bash
cmake -S . -B build
cmake --build build --config Release
```

### D. g++ / clang++ を直接使う

```bash
g++ -O2 -std=c++17 -o lecun1989 lecun1989.cpp
g++ -O2 -std=c++17 -o predict  predict.cpp stb_impl.cpp
# （clang++ を使う場合は g++ を置き換えてください）
```

---

## 2. まず正しさを確認（データ不要）

手書きの逆伝播を数値微分と突き合わせて検証します。データセットは不要です。

```
lecun1989 gradcheck
```

出力例:

```
gradient check (analytic vs numerical, eps=1e-06):
  H1w   max rel err = 7.226e-09
  ...
overall max rel err = 1.712e-07  ->  PASS
```

最大相対誤差が 1e-5 未満なら `PASS` と表示され、勾配が導出どおり正確に
実装されていることを確認できます。

---

## 3. データの入手（MNIST）

1989 年当時の正確なデータセットは入手できないため、`prepro.py` と同様に
MNIST をランダムサンプリングして 16x16 に縮小することで近似します
（訓練 7291 枚 / テスト 2007 枚、画素は [-1, 1]、クラス目標は ±1）。

**展開済み**の MNIST IDX ファイル 4 つを `data/` フォルダに置くだけです。

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

### Windows（PowerShell、外部ツール不要）

```powershell
powershell -ExecutionPolicy Bypass -File get_mnist.ps1
```

### macOS / Linux / Git Bash

```bash
mkdir -p data && cd data
base=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sSL "$base/$f.gz" | gzip -d > "$f"
done
```

> ドット区切りの命名（`train-images.idx3-ubyte`）も認識されます。

---

## 4. 学習

```
lecun1989 train [data_dir] [learning_rate] [--epochs N] [--save FILE]
```

デフォルト: データディレクトリ `data`、学習率 `0.03`、23 パス（論文と同じ）。
`--save FILE` で学習済みの重みを書き出すと、後で推論に使えます。

```
lecun1989 train                          # デフォルト
lecun1989 train data 0.03 --save model.bin
```

最終パスの例（このリポジトリでの実測）:

```
23
eval: split train. loss 9.035124e-03. error 1.06%. misses: 77
eval: split test . loss 3.274319e-02. error 4.83%. misses: 97
```

同じネットワークの他実装との比較:

| | 訓練誤差 | テスト誤差 | テスト誤判定 |
|---|---|---|---|
| 論文 (1989) | 0.14% | 5.00% | 102 |
| repro.py (PyTorch) | 0.62% | 4.09% | 82 |
| **この C++ 移植版** | **1.06%** | **4.83%** | **97** |

数値が完全には一致しないのは、1989 年のデータセットを MNIST で近似している
こと、また乱数（サンプリング順）やリサイズの補間が Python とビット単位では
一致しないためです。元の README にあるとおり「この差異の大部分は訓練データ
そのものに由来する」ものです。ネットワークと学習アルゴリズム自体は忠実に
再現されています。

最適化ビルドなら 23 パスすべてが 1 分足らずで終わります
（1989 年当時は SUN-4 で 3 日かかりました）。

---

## 5. 保存済みモデルからの推論

```
lecun1989 eval [data_dir] --load FILE
```

`--save` で書き出した重みを読み込み、テストセットの誤差だけを報告します
（学習はしません）。

```
lecun1989 eval data --load model.bin
```

---

## 6. 画像ファイルから数字を認識する

`predict` は `--save` で保存したモデルを読み込み、通常の画像
（PNG / JPG / BMP / …）に写った手書き数字 1 文字を分類します。
画像の読み込みには [`stb_image`](https://github.com/nothings/stb)、
16×16 への縮小には `stb_image_resize2` を使います（どちらもヘッダオンリーで、
このフォルダに同梱済み。実装は `stb_impl.cpp` で 1 回だけコンパイルされます）。

```
predict <image> [--model FILE] [--invert|--no-invert] [--no-center]
```

| オプション | 意味 |
|---|---|
| `<image>` | 数字画像のパス（stb_image が読める形式なら何でも） |
| `--model FILE` | 読み込むモデルファイル（デフォルト `model.bin`） |
| `--invert` / `--no-invert` | 色反転を強制 / 無効化（デフォルト: 自動判定） |
| `--no-center` | バウンディングボックスの切り出し・中央寄せをせず、そのまま 16×16 に縮小 |

**学習済みモデルを同梱**しています（このフォルダの `model.bin`、23 エポック、
MNIST テスト誤差 約 4.5%）。学習せずにすぐ `predict` を試せます。
`lecun1989 train ... --save model.bin` でいつでも再学習できます。

このネットは MNIST 形式、つまり **暗い背景に明るい数字**で学習されています。
紙にペンで書いたものを撮影・スキャンした画像は逆（明るい紙に暗いインク）なので、
`predict` は画像の枠が明るいときにデフォルトで自動反転します。さらに数字の
バウンディングボックスに切り出し、MNIST 風の余白を付けて中央寄せするため、
実世界の写真でもかなり頑健になります。自動判定を上書きしたいときは
`--invert` / `--no-invert` を使ってください。

```
> predict 2ni.png --model model.bin
loaded model 'model.bin' and image '2ni.png' (100x100, 4 channel(s))
colour inversion: on (auto)

  predicted digit: 2

  per-class scores (tanh output, higher = more likely):
    2: +0.9964  ########################################  <-- prediction
    3: -0.9871
    ...
```

精度を上げるコツ: そこそこ中央に寄った 1 文字で、コントラストの良い画像を
使ってください。1989 年のネットはとても小さく（入力 16×16、9760 パラメータ）、
縮小した MNIST で学習しているため、極端に崩れた字やノイズの多い背景では
外すことがあります。

---

## 7. 学習率スイープ

論文には学習率の記載がなく、原著メモでは手動での探索に触れられています。
このモードは学習率ごとに（公平な比較のため同一の初期化から）新しいネットを
学習し、最良のものを報告します。

```
lecun1989 sweep [data_dir] [--epochs N] [lr1 lr2 ...]
```

デフォルト: 学習率 `0.003 0.01 0.03 0.1 0.3`、各 10 エポック
（スイープを現実的な時間に収めるため、フル学習より少なめ）。例:

```
lecun1989 sweep data --epochs 10
```

```
lr           test err%    misses     test loss
------------------------------------------------
0.003        ...
0.03         ...                      <-- best
...
best learning rate: 0.03 (test error ...%)
```

---

## 実装メモ

- 演算は `double`（数値的安定性と、きれいな勾配チェックのため。ネットワーク
  構造自体は浮動小数点精度に依存しません）。
- 12→8 の H2 疎結合は Karpathy 版と同じブロック構造: 出力面 0–3 は入力面 0–7、
  面 4–7 は 4–11、面 8–11 は 0–3 と 8–11（「クロス」グループ）を読みます。
- 損失は 10 出力にわたる `mean((y - yhat)^2)`。`1/10` の係数は勾配にも反映。
- 重み初期化は `U(-2.4/sqrt(fan_in), +2.4/sqrt(fan_in))`。出力バイアスだけ
  -1 から始めます。
- モデルファイルは小さなバイナリ形式（マジックタグ、バージョン、パラメータ数、
  続いて全パラメータをリトルエンディアンの double で並べたもの）。

## コマンド一覧

| コマンド | 用途 |
|---|---|
| `lecun1989 gradcheck` | 逆伝播を数値的に検証（データ不要） |
| `lecun1989 train [dir] [lr] [--epochs N] [--save FILE]` | 学習し、必要なら重みを保存 |
| `lecun1989 eval [dir] --load FILE` | モデルを読み込みテスト誤差を報告 |
| `lecun1989 sweep [dir] [--epochs N] [lrs...]` | 学習率を比較して最良を報告 |
| `predict <image> [--model FILE] [--invert\|--no-invert] [--no-center]` | 画像内の数字を認識 |

## ファイル構成

| ファイル | 用途 |
|---|---|
| `lecun_net.h` | ネットワーク本体（次元・結合・順/逆伝播・SGD・モデルの保存/読込）— 共有 |
| `lecun1989.cpp` | 学習・評価 CLI（`gradcheck`, `train`, `eval`, `sweep`） |
| `predict.cpp` | 単体の画像分類プログラム |
| `stb_image.h`, `stb_image_resize2.h`, `stb_impl.cpp` | `predict` 用に同梱した stb ライブラリ |
| `model.bin` | 同梱の学習済みモデル |
| `CMakeLists.txt` | `lecun1989` と `predict` の両方をビルド |
