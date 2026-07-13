# lecun1989-repro — C/C++ port

A dependency-free **standard C++17** port of Andrej Karpathy's reproduction of
Yann LeCun et al. 1989,
[Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
— arguably the first real-world neural network trained with backpropagation.

It ports the original [`repro.py`](../repro.py) / [`prepro.py`](../prepro.py)
(PyTorch) to a single self-contained C++ file.

- **No PyTorch, no BLAS, no third-party libraries** — only the C++ standard library.
- Builds and runs unchanged with **MSVC (Visual Studio 2022), GCC and Clang**, on
  Windows, Linux and macOS.
- Convolutions, the sparse H2 connectivity (8-of-12 planes), the per-position
  (unshared) biases, tanh activations on every layer, the mean-squared-error
  loss, the paper's weight initialisation and plain SGD are all re-implemented
  **by hand, forward AND backward**.
- Parameter count is exactly **9760**, matching the paper.

The only source file is `lecun1989.cpp` (plus `CMakeLists.txt` and helper scripts).

---

## 1. Build

### A. Visual Studio 2022 (CMake — easiest)

VS2022 opens CMake projects directly.

1. Launch Visual Studio 2022 → **Open a Local Folder** → select this `cpp` folder.
2. VS auto-detects `CMakeLists.txt`. Switch the configuration to **`x64-Release`**.
3. **Build → Build All**.
4. You get `lecun1989.exe`.

### B. Visual Studio 2022 (command line)

Open the **x64 Native Tools Command Prompt for VS 2022**:

```bat
cl /O2 /EHsc /std:c++17 lecun1989.cpp
```

### C. CMake (any compiler)

```bash
cmake -S . -B build
cmake --build build --config Release
```

### D. g++ / clang++ directly

```bash
g++ -O2 -std=c++17 -o lecun1989 lecun1989.cpp
# or
clang++ -O2 -std=c++17 -o lecun1989 lecun1989.cpp
```

---

## 2. Verify correctness first (no data required)

Check the hand-written backprop against numerical gradients — no dataset needed:

```
lecun1989 gradcheck
```

Example output:

```
gradient check (analytic vs numerical, eps=1e-06):
  H1w   max rel err = 7.226e-09
  ...
overall max rel err = 1.712e-07  ->  PASS
```

A max relative error below 1e-5 prints `PASS`, confirming the gradients are
implemented exactly as derived.

---

## 3. Get the data (MNIST)

The exact 1989 dataset is unavailable, so — like `prepro.py` — we approximate it
by randomly sampling MNIST and downscaling to 16x16 (7291 train / 2007 test
digits, pixels in [-1, 1], class targets ±1).

You just need the four **decompressed** MNIST IDX files in a `data/` folder:

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

### Windows (PowerShell, no external tools)

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

> The dot-separated naming (`train-images.idx3-ubyte`) is also recognised.

---

## 4. Train

```
lecun1989 train [data_dir] [learning_rate] [--epochs N] [--save FILE]
```

Defaults: data dir `data`, learning rate `0.03`, 23 passes (as in the paper).
`--save FILE` writes the trained weights so you can run inference later.

```
lecun1989 train                          # defaults
lecun1989 train data 0.03 --save model.bin
```

Example final pass (measured in this repo):

```
23
eval: split train. loss 9.035124e-03. error 1.06%. misses: 77
eval: split test . loss 3.274319e-02. error 4.83%. misses: 97
```

Comparison with other implementations of the same network:

| | train error | test error | test misses |
|---|---|---|---|
| Paper, 1989 | 0.14% | 5.00% | 102 |
| repro.py (PyTorch) | 0.62% | 4.09% | 82 |
| **this C++ port** | **1.06%** | **4.83%** | **97** |

The numbers don't match exactly because the 1989 dataset is only approximated
with MNIST, and the RNG (sampling order) and the resize interpolation don't
match Python bit-for-bit. As the original README notes, "the majority of this
discrepancy comes from the training dataset itself." The network and training
algorithm themselves are reproduced faithfully.

An optimised build finishes all 23 passes in well under a minute (the original
took 3 days on a SUN-4 in 1989).

---

## 5. Inference from a saved model

```
lecun1989 eval [data_dir] --load FILE
```

Loads the weights written by `--save` and reports test-set error only (no
training):

```
lecun1989 eval data --load model.bin
```

---

## 6. Learning-rate sweep

The paper never states the learning rate; the original notes mention a manual
sweep. This mode trains one fresh net per learning rate (from an identical
initialisation, for a fair comparison) and reports the best:

```
lecun1989 sweep [data_dir] [--epochs N] [lr1 lr2 ...]
```

Defaults: learning rates `0.003 0.01 0.03 0.1 0.3`, 10 epochs each (fewer than
a full run, to keep the sweep practical). Example:

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

## Implementation notes

- Uses `double` arithmetic (for numerical stability and clean gradient checking;
  the network structure does not depend on the floating-point precision).
- The 12→8 sparse H2 connectivity uses the same block structure as Karpathy's
  version: output planes 0–3 read input planes 0–7, planes 4–7 read 4–11, and
  planes 8–11 read 0–3 and 8–11 (the "cross" group).
- Loss is `mean((y - yhat)^2)` over the 10 outputs; the `1/10` factor is carried
  through the gradients.
- Weight init is `U(-2.4/sqrt(fan_in), +2.4/sqrt(fan_in))`; only the output
  biases start at -1.
- Model files use a small binary format (a magic tag, version, param count,
  then all parameters as little-endian doubles).

## Command reference

| Command | Purpose |
|---|---|
| `lecun1989 gradcheck` | Verify backprop numerically (no data needed) |
| `lecun1989 train [dir] [lr] [--epochs N] [--save FILE]` | Train, optionally save weights |
| `lecun1989 eval [dir] --load FILE` | Load a model and report test error |
| `lecun1989 sweep [dir] [--epochs N] [lrs...]` | Compare learning rates, report the best |
