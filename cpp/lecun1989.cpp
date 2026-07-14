// ---------------------------------------------------------------------------
//  lecun1989.cpp
//
//  A faithful, dependency-free C/C++ port of Andrej Karpathy's reproduction of
//  Yann LeCun et al. 1989 "Backpropagation Applied to Handwritten Zip Code
//  Recognition" -- arguably the first real-world neural net trained with
//  backpropagation.
//
//  Original Python (PyTorch) reference: repro.py / prepro.py in this repo.
//
//  This file is self-contained standard C++17. No PyTorch, no BLAS, no third
//  party libraries -- only the C++ standard library. It therefore builds and
//  runs unchanged on MSVC (Visual Studio 2022), GCC and Clang, on Windows,
//  Linux and macOS.
//
//  The convolutions, the sparse H2 connectivity, the per-position (unshared)
//  biases, the tanh activations, the mean-squared-error loss, the paper's
//  weight initialisation and plain SGD are all re-implemented by hand, forward
//  AND backward.
//
//  Usage:
//     lecun1989 gradcheck
//         Verify the hand-written backprop against numerical gradients.
//         Needs no data -- great for confirming correctness anywhere.
//
//     lecun1989 train [data_dir] [learning_rate]
//         Load MNIST (raw IDX ubyte files) from data_dir (default "data"),
//         build the 1989 dataset (7291 train / 2007 test, 16x16), and train
//         for 23 passes, printing train/test loss + error after each pass.
//         learning_rate default 0.03 (as in repro.py).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>

// The network itself (dimensions, connectivity, Net struct, save/load) lives in
// a shared header so the trainer and the image classifier stay in lock-step.
#include "lecun_net.h"

#if 0  // ---- moved to lecun_net.h ----
// ===========================================================================
//  Network dimensions (see repro.py for the derivation of every number)
// ===========================================================================
namespace dim {
    // input image
    constexpr int IN = 16;              // 16x16 input
    // H1: 5x5 stride-2 conv, 12 planes, input padded by 2 with -1  -> 8x8x12
    constexpr int P1 = 20;              // padded input side (16 + 2*2)
    constexpr int H1C = 12, H1S = 8;    // channels, spatial side
    // H2: sparse 5x5 stride-2 conv, 12 planes, a1 padded by 2 -> 4x4x12
    constexpr int P2 = 12;              // padded a1 side (8 + 2*2)
    constexpr int H2C = 12, H2S = 4;
    // fully connected layers
    constexpr int FLAT = H2C * H2S * H2S; // 192
    constexpr int H3 = 30;
    constexpr int OUT = 10;

    // parameter tensor element counts
    constexpr int nH1w = H1C * 1 * 5 * 5;   // 300
    constexpr int nH1b = H1C * H1S * H1S;   // 768
    constexpr int nH2w = H2C * 8 * 5 * 5;   // 2400  (each out plane reads 8 in planes)
    constexpr int nH2b = H2C * H2S * H2S;   // 192
    constexpr int nH3w = FLAT * H3;         // 5760
    constexpr int nH3b = H3;                // 30
    constexpr int nOutw = H3 * OUT;         // 300
    constexpr int nOutb = OUT;              // 10
    constexpr int nParams =
        nH1w + nH1b + nH2w + nH2b + nH3w + nH3b + nOutw + nOutb; // 9760
}

// Which physical input plane does weight-slot j (0..7) of output plane `co`
// read from?  This is Karpathy's sensible block connectivity for the paper's
// unspecified 12->8 scheme:
//   out planes 0..3  read input planes 0..7
//   out planes 4..7  read input planes 4..11
//   out planes 8..11 read input planes 0..3 and 8..11 (the "cross" group)
static inline int inplane(int co, int j) {
    int group = co / 4;
    if (group == 0) return j;             // 0..7
    if (group == 1) return j + 4;         // 4..11
    return (j < 4) ? j : (j + 4);         // 0,1,2,3, 8,9,10,11
}

// ===========================================================================
//  The 1989 ConvNet
// ===========================================================================
struct Net {
    // ---- parameters ----
    std::vector<double> H1w, H1b, H2w, H2b, H3w, H3b, outw, outb;
    // ---- gradients (same shapes) ----
    std::vector<double> gH1w, gH1b, gH2w, gH2b, gH3w, gH3b, gOutw, gOutb;

    // ---- forward-pass activation caches ----
    std::array<double, dim::P1 * dim::P1>            xp;   // padded input (1 chan)
    std::array<double, dim::H1C * dim::H1S * dim::H1S> a1; // tanh(H1)
    std::array<double, dim::H2C * dim::P2 * dim::P2>  ap;  // padded a1
    std::array<double, dim::FLAT>                     a2;  // tanh(H2), also flat feature
    std::array<double, dim::H3>                       a3;  // tanh(H3)
    std::array<double, dim::OUT>                      yhat;// tanh(out)

    Net() { alloc(); }

    void alloc() {
        H1w.assign(dim::nH1w, 0); H1b.assign(dim::nH1b, 0);
        H2w.assign(dim::nH2w, 0); H2b.assign(dim::nH2b, 0);
        H3w.assign(dim::nH3w, 0); H3b.assign(dim::nH3b, 0);
        outw.assign(dim::nOutw, 0); outb.assign(dim::nOutb, 0);
        gH1w.assign(dim::nH1w, 0); gH1b.assign(dim::nH1b, 0);
        gH2w.assign(dim::nH2w, 0); gH2b.assign(dim::nH2b, 0);
        gH3w.assign(dim::nH3w, 0); gH3b.assign(dim::nH3b, 0);
        gOutw.assign(dim::nOutw, 0); gOutb.assign(dim::nOutb, 0);
    }

    // Paper's initialisation: w ~ U(-a, a) with a = 2.4 / sqrt(fan_in).
    // Biases: H1/H2/H3 to zero, output bias to -1 (9/10 targets are -1).
    void init(uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        auto winit = [&](std::vector<double>& w, double fan_in) {
            double a = 2.4 / std::sqrt(fan_in);
            for (double& v : w) v = (u(rng) - 0.5) * 2.0 * a;
        };
        winit(H1w, 5 * 5 * 1);
        winit(H2w, 5 * 5 * 8);
        winit(H3w, dim::FLAT);
        winit(outw, dim::H3);
        std::fill(H1b.begin(), H1b.end(), 0.0);
        std::fill(H2b.begin(), H2b.end(), 0.0);
        std::fill(H3b.begin(), H3b.end(), 0.0);
        std::fill(outb.begin(), outb.end(), -1.0);
    }

    // ---- forward pass. x = 16*16 doubles in [-1,1]. Fills caches, returns yhat. ----
    void forward(const double* x) {
        using namespace dim;

        // pad input with -1 background
        std::fill(xp.begin(), xp.end(), -1.0);
        for (int y = 0; y < IN; ++y)
            for (int xx = 0; xx < IN; ++xx)
                xp[(y + 2) * P1 + (xx + 2)] = x[y * IN + xx];

        // H1: conv 5x5 stride 2, per-position bias, tanh
        for (int co = 0; co < H1C; ++co)
            for (int oy = 0; oy < H1S; ++oy)
                for (int ox = 0; ox < H1S; ++ox) {
                    double acc = H1b[(co * H1S + oy) * H1S + ox];
                    for (int ky = 0; ky < 5; ++ky)
                        for (int kx = 0; kx < 5; ++kx)
                            acc += xp[(oy * 2 + ky) * P1 + (ox * 2 + kx)]
                                 * H1w[co * 25 + ky * 5 + kx];
                    a1[(co * H1S + oy) * H1S + ox] = std::tanh(acc);
                }

        // pad a1 with -1 background
        std::fill(ap.begin(), ap.end(), -1.0);
        for (int c = 0; c < H1C; ++c)
            for (int y = 0; y < H1S; ++y)
                for (int xx = 0; xx < H1S; ++xx)
                    ap[(c * P2 + (y + 2)) * P2 + (xx + 2)] =
                        a1[(c * H1S + y) * H1S + xx];

        // H2: sparse conv 5x5 stride 2, per-position bias, tanh
        for (int co = 0; co < H2C; ++co)
            for (int oy = 0; oy < H2S; ++oy)
                for (int ox = 0; ox < H2S; ++ox) {
                    double acc = H2b[(co * H2S + oy) * H2S + ox];
                    for (int j = 0; j < 8; ++j) {
                        int ip = inplane(co, j);
                        for (int ky = 0; ky < 5; ++ky)
                            for (int kx = 0; kx < 5; ++kx)
                                acc += ap[(ip * P2 + (oy * 2 + ky)) * P2 + (ox * 2 + kx)]
                                     * H2w[((co * 8 + j) * 5 + ky) * 5 + kx];
                    }
                    a2[(co * H2S + oy) * H2S + ox] = std::tanh(acc);
                }

        // H3: fully connected, tanh.  a2 (192) is the flat feature vector.
        for (int o = 0; o < H3; ++o) {
            double acc = H3b[o];
            for (int i = 0; i < FLAT; ++i) acc += a2[i] * H3w[i * H3 + o];
            a3[o] = std::tanh(acc);
        }

        // output: fully connected, tanh
        for (int o = 0; o < OUT; ++o) {
            double acc = outb[o];
            for (int i = 0; i < H3; ++i) acc += a3[i] * outw[i * OUT + o];
            yhat[o] = std::tanh(acc);
        }
    }

    // mean squared error over the 10 outputs (matches torch.mean((y-yhat)**2))
    double loss(const double* y) const {
        double s = 0;
        for (int o = 0; o < dim::OUT; ++o) { double d = y[o] - yhat[o]; s += d * d; }
        return s / dim::OUT;
    }

    void zero_grad() {
        std::fill(gH1w.begin(), gH1w.end(), 0.0); std::fill(gH1b.begin(), gH1b.end(), 0.0);
        std::fill(gH2w.begin(), gH2w.end(), 0.0); std::fill(gH2b.begin(), gH2b.end(), 0.0);
        std::fill(gH3w.begin(), gH3w.end(), 0.0); std::fill(gH3b.begin(), gH3b.end(), 0.0);
        std::fill(gOutw.begin(), gOutw.end(), 0.0); std::fill(gOutb.begin(), gOutb.end(), 0.0);
    }

    // ---- backward pass. Assumes forward() was just called with matching x. ----
    void backward(const double* y) {
        using namespace dim;
        zero_grad();

        // d loss / d yhat, then through output tanh
        std::array<double, OUT> dz4;
        for (int o = 0; o < OUT; ++o) {
            double dyhat = 2.0 / OUT * (yhat[o] - y[o]);
            dz4[o] = dyhat * (1.0 - yhat[o] * yhat[o]);
        }
        // output weights/bias, and grad wrt a3
        std::array<double, H3> da3; da3.fill(0.0);
        for (int i = 0; i < H3; ++i)
            for (int o = 0; o < OUT; ++o) {
                gOutw[i * OUT + o] += a3[i] * dz4[o];
                da3[i] += dz4[o] * outw[i * OUT + o];
            }
        for (int o = 0; o < OUT; ++o) gOutb[o] += dz4[o];

        // through H3 tanh
        std::array<double, H3> dz3;
        for (int o = 0; o < H3; ++o) dz3[o] = da3[o] * (1.0 - a3[o] * a3[o]);
        // H3 weights/bias, and grad wrt flat feature (== a2)
        std::array<double, FLAT> da2; da2.fill(0.0);
        for (int i = 0; i < FLAT; ++i)
            for (int o = 0; o < H3; ++o) {
                gH3w[i * H3 + o] += a2[i] * dz3[o];
                da2[i] += dz3[o] * H3w[i * H3 + o];
            }
        for (int o = 0; o < H3; ++o) gH3b[o] += dz3[o];

        // through H2 tanh
        std::array<double, FLAT> dz2;
        for (int i = 0; i < FLAT; ++i) dz2[i] = da2[i] * (1.0 - a2[i] * a2[i]);

        // backprop sparse conv H2 -> grads for H2w/H2b and grad wrt padded a1
        std::vector<double> dap(H2C * P2 * P2, 0.0);
        for (int co = 0; co < H2C; ++co)
            for (int oy = 0; oy < H2S; ++oy)
                for (int ox = 0; ox < H2S; ++ox) {
                    double d = dz2[(co * H2S + oy) * H2S + ox];
                    gH2b[(co * H2S + oy) * H2S + ox] += d;
                    for (int j = 0; j < 8; ++j) {
                        int ip = inplane(co, j);
                        for (int ky = 0; ky < 5; ++ky)
                            for (int kx = 0; kx < 5; ++kx) {
                                int aidx = (ip * P2 + (oy * 2 + ky)) * P2 + (ox * 2 + kx);
                                int widx = ((co * 8 + j) * 5 + ky) * 5 + kx;
                                gH2w[widx] += ap[aidx] * d;
                                dap[aidx] += H2w[widx] * d;
                            }
                    }
                }

        // crop padding -> grad wrt a1, then through H1 tanh
        std::array<double, H1C * H1S * H1S> dz1;
        for (int c = 0; c < H1C; ++c)
            for (int y2 = 0; y2 < H1S; ++y2)
                for (int x2 = 0; x2 < H1S; ++x2) {
                    double da1 = dap[(c * P2 + (y2 + 2)) * P2 + (x2 + 2)];
                    double a = a1[(c * H1S + y2) * H1S + x2];
                    dz1[(c * H1S + y2) * H1S + x2] = da1 * (1.0 - a * a);
                }

        // backprop conv H1 -> grads for H1w/H1b (no need to propagate to input)
        for (int co = 0; co < H1C; ++co)
            for (int oy = 0; oy < H1S; ++oy)
                for (int ox = 0; ox < H1S; ++ox) {
                    double d = dz1[(co * H1S + oy) * H1S + ox];
                    gH1b[(co * H1S + oy) * H1S + ox] += d;
                    for (int ky = 0; ky < 5; ++ky)
                        for (int kx = 0; kx < 5; ++kx)
                            gH1w[co * 25 + ky * 5 + kx] +=
                                xp[(oy * 2 + ky) * dim::P1 + (ox * 2 + kx)] * d;
                }
    }

    // plain SGD step: p -= lr * grad
    void sgd_step(double lr) {
        auto step = [&](std::vector<double>& p, const std::vector<double>& g) {
            for (size_t i = 0; i < p.size(); ++i) p[i] -= lr * g[i];
        };
        step(H1w, gH1w); step(H1b, gH1b);
        step(H2w, gH2w); step(H2b, gH2b);
        step(H3w, gH3w); step(H3b, gH3b);
        step(outw, gOutw); step(outb, gOutb);
    }

    int argmax_yhat() const {
        int best = 0;
        for (int o = 1; o < dim::OUT; ++o) if (yhat[o] > yhat[best]) best = o;
        return best;
    }

    // ---- model persistence ---------------------------------------------------
    // Simple binary format: magic "L89M", version, param count, then every
    // parameter tensor's doubles in a fixed order. Little-endian (x86/ARM).
    static const uint32_t MAGIC = 0x4D393845; // "E89M" bytes -> arbitrary tag
    static const uint32_t VERSION = 1;

    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = MAGIC, ver = VERSION, np = dim::nParams;
        f.write(reinterpret_cast<const char*>(&magic), 4);
        f.write(reinterpret_cast<const char*>(&ver), 4);
        f.write(reinterpret_cast<const char*>(&np), 4);
        auto wr = [&](const std::vector<double>& v) {
            f.write(reinterpret_cast<const char*>(v.data()),
                    (std::streamsize)(v.size() * sizeof(double)));
        };
        wr(H1w); wr(H1b); wr(H2w); wr(H2b);
        wr(H3w); wr(H3b); wr(outw); wr(outb);
        return (bool)f;
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = 0, ver = 0, np = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&np), 4);
        if (magic != MAGIC || ver != VERSION || np != (uint32_t)dim::nParams) {
            std::fprintf(stderr, "ERROR: '%s' is not a compatible model file\n", path.c_str());
            return false;
        }
        auto rd = [&](std::vector<double>& v) {
            f.read(reinterpret_cast<char*>(v.data()),
                   (std::streamsize)(v.size() * sizeof(double)));
        };
        rd(H1w); rd(H1b); rd(H2w); rd(H2b);
        rd(H3w); rd(H3b); rd(outw); rd(outb);
        return (bool)f;
    }
};
#endif // ---- moved to lecun_net.h ----

// ===========================================================================
//  Gradient check -- verifies backward() against numerical gradients.
// ===========================================================================
static int run_gradcheck() {
    Net net; net.init(1337);

    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::array<double, dim::IN * dim::IN> x;
    for (double& v : x) v = u(rng);
    std::array<double, dim::OUT> y; y.fill(-1.0);
    y[std::uniform_int_distribution<int>(0, 9)(rng)] = 1.0;

    // analytic gradients
    net.forward(x.data());
    net.backward(y.data());

    // Check a spread of parameters across every tensor.
    struct Grp { const char* name; std::vector<double>* p; std::vector<double>* g; };
    std::vector<Grp> groups = {
        {"H1w", &net.H1w, &net.gH1w}, {"H1b", &net.H1b, &net.gH1b},
        {"H2w", &net.H2w, &net.gH2w}, {"H2b", &net.H2b, &net.gH2b},
        {"H3w", &net.H3w, &net.gH3w}, {"H3b", &net.H3b, &net.gH3b},
        {"outw", &net.outw, &net.gOutw}, {"outb", &net.outb, &net.gOutb},
    };

    const double eps = 1e-6;
    double max_rel = 0.0;
    std::printf("gradient check (analytic vs numerical, eps=%.0e):\n", eps);
    for (auto& gr : groups) {
        double grp_max = 0.0;
        int n = (int)gr.p->size();
        int stride = std::max(1, n / 17);   // sample ~17 params per tensor
        for (int i = 0; i < n; i += stride) {
            double orig = (*gr.p)[i];
            (*gr.p)[i] = orig + eps; net.forward(x.data()); double lp = net.loss(y.data());
            (*gr.p)[i] = orig - eps; net.forward(x.data()); double lm = net.loss(y.data());
            (*gr.p)[i] = orig;
            double num = (lp - lm) / (2 * eps);
            double ana = (*gr.g)[i];
            double denom = std::max(1e-12, std::fabs(num) + std::fabs(ana));
            double rel = std::fabs(num - ana) / denom;
            grp_max = std::max(grp_max, rel);
        }
        max_rel = std::max(max_rel, grp_max);
        std::printf("  %-5s max rel err = %.3e\n", gr.name, grp_max);
    }
    std::printf("overall max rel err = %.3e  ->  %s\n",
                max_rel, max_rel < 1e-5 ? "PASS" : "FAIL");
    return max_rel < 1e-5 ? 0 : 1;
}

// ===========================================================================
//  MNIST loading + 1989-style preprocessing (mirrors prepro.py)
// ===========================================================================
struct Example { std::array<double, dim::IN * dim::IN> x; int label; };

static uint32_t read_be32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 |
           (uint32_t)b[2] << 8  | (uint32_t)b[3];
}

static std::ifstream open_first(const std::string& dir,
                                const std::vector<std::string>& names) {
    for (const auto& n : names) {
        std::ifstream f(dir + "/" + n, std::ios::binary);
        if (f) return f;
    }
    return std::ifstream(); // not open
}

// Bilinear resize 28x28 -> 16x16, align_corners=False (as torch F.interpolate).
static void resize_28_to_16(const std::vector<float>& src /*784, in [-1,1]*/,
                            std::array<double, dim::IN * dim::IN>& dst) {
    const int SS = 28, DS = dim::IN;
    const double scale = (double)SS / DS; // 1.75
    for (int oy = 0; oy < DS; ++oy) {
        double sy = (oy + 0.5) * scale - 0.5; if (sy < 0) sy = 0;
        int y0 = (int)std::floor(sy); int y1 = std::min(y0 + 1, SS - 1);
        double wy = sy - y0;
        for (int ox = 0; ox < DS; ++ox) {
            double sx = (ox + 0.5) * scale - 0.5; if (sx < 0) sx = 0;
            int x0 = (int)std::floor(sx); int x1 = std::min(x0 + 1, SS - 1);
            double wx = sx - x0;
            double v00 = src[y0 * SS + x0], v01 = src[y0 * SS + x1];
            double v10 = src[y1 * SS + x0], v11 = src[y1 * SS + x1];
            double top = v00 * (1 - wx) + v01 * wx;
            double bot = v10 * (1 - wx) + v11 * wx;
            dst[oy * DS + ox] = top * (1 - wy) + bot * wy;
        }
    }
}

// Load `n` random examples from an MNIST split, preprocessed to the 1989 format.
static bool load_split(const std::string& dir, bool train, int n,
                       std::vector<Example>& out, uint32_t seed) {
    std::ifstream fi = open_first(dir, train
        ? std::vector<std::string>{"train-images-idx3-ubyte", "train-images.idx3-ubyte"}
        : std::vector<std::string>{"t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"});
    std::ifstream fl = open_first(dir, train
        ? std::vector<std::string>{"train-labels-idx1-ubyte", "train-labels.idx1-ubyte"}
        : std::vector<std::string>{"t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"});
    if (!fi || !fl) {
        std::fprintf(stderr, "ERROR: could not open MNIST %s files in '%s'.\n",
                     train ? "train" : "test", dir.c_str());
        return false;
    }
    if (read_be32(fi) != 2051) { std::fprintf(stderr, "bad image magic\n"); return false; }
    uint32_t nimg = read_be32(fi), rows = read_be32(fi), cols = read_be32(fi);
    if (read_be32(fl) != 2049) { std::fprintf(stderr, "bad label magic\n"); return false; }
    uint32_t nlab = read_be32(fl);
    if (rows != 28 || cols != 28 || nimg != nlab) {
        std::fprintf(stderr, "unexpected MNIST dims\n"); return false;
    }

    // read all raw images and labels
    std::vector<unsigned char> pix((size_t)nimg * 784), lab(nlab);
    fi.read(reinterpret_cast<char*>(pix.data()), pix.size());
    fl.read(reinterpret_cast<char*>(lab.data()), lab.size());

    // random permutation, take first n (mirrors np.random.permutation(len)[:n])
    std::vector<int> perm(nimg);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    for (int i = (int)nimg - 1; i > 0; --i)   // Fisher-Yates
        std::swap(perm[i], perm[std::uniform_int_distribution<int>(0, i)(rng)]);

    out.clear(); out.reserve(n);
    std::vector<float> img(784);
    for (int k = 0; k < n; ++k) {
        int ix = perm[k];
        const unsigned char* p = &pix[(size_t)ix * 784];
        for (int i = 0; i < 784; ++i) img[i] = (float)p[i] / 127.5f - 1.0f; // [-1,1]
        Example ex;
        resize_28_to_16(img, ex.x);
        ex.label = lab[ix];
        out.push_back(ex);
    }
    return true;
}

// ===========================================================================
//  Training (mirrors repro.py)
// ===========================================================================
struct EvalResult { double loss; double err; int misses; };

static EvalResult eval_split(Net& net, const std::vector<Example>& data) {
    double loss_sum = 0.0; int misses = 0;
    double y[dim::OUT];
    for (const auto& ex : data) {
        for (int o = 0; o < dim::OUT; ++o) y[o] = -1.0;
        y[ex.label] = 1.0;
        net.forward(ex.x.data());
        loss_sum += net.loss(y);
        if (net.argmax_yhat() != ex.label) ++misses;
    }
    EvalResult r;
    r.loss = loss_sum / data.size();
    r.misses = misses;
    r.err = (double)misses / data.size();
    return r;
}

static void print_eval(const char* name, const EvalResult& r) {
    std::printf("eval: split %-5s. loss %e. error %.2f%%. misses: %d\n",
                name, r.loss, r.err * 100.0, r.misses);
}

// Train a fresh (already-initialised) net. Returns final test-set metrics.
// If verbose, prints per-pass train/test metrics like repro.py.
static EvalResult train_net(Net& net, const std::vector<Example>& Xtr,
                            const std::vector<Example>& Xte,
                            double lr, int epochs, bool verbose) {
    double y[dim::OUT];
    EvalResult te{};
    for (int pass_num = 0; pass_num < epochs; ++pass_num) {
        // one epoch of SGD, one example at a time, in fixed order
        for (const auto& ex : Xtr) {
            for (int o = 0; o < dim::OUT; ++o) y[o] = -1.0;
            y[ex.label] = 1.0;
            net.forward(ex.x.data());
            net.backward(y);
            net.sgd_step(lr);
        }
        te = eval_split(net, Xte);
        if (verbose) {
            EvalResult tr = eval_split(net, Xtr);
            std::printf("%d\n", pass_num + 1);
            print_eval("train", tr);
            print_eval("test",  te);
            std::fflush(stdout);
        }
    }
    return te;
}

static int run_train(const std::string& dir, double lr, int epochs,
                     const std::string& save_path) {
    std::printf("learning rate: %g, epochs: %d, data dir: %s\n", lr, epochs, dir.c_str());

    std::vector<Example> Xtr, Xte;
    if (!load_split(dir, true,  7291, Xtr, 1337)) return 1;
    if (!load_split(dir, false, 2007, Xte, 1338)) return 1;
    std::printf("loaded %zu train, %zu test examples\n", Xtr.size(), Xte.size());

    Net net; net.init(1337);
    std::printf("# params: %d (paper: 9760)\n", dim::nParams);

    train_net(net, Xtr, Xte, lr, epochs, /*verbose=*/true);

    if (!save_path.empty()) {
        if (net.save(save_path))
            std::printf("saved model to %s\n", save_path.c_str());
        else
            std::fprintf(stderr, "ERROR: could not write model to %s\n", save_path.c_str());
    }
    return 0;
}

// Load a trained model and evaluate it on the test set (inference only).
static int run_eval(const std::string& dir, const std::string& load_path) {
    if (load_path.empty()) {
        std::fprintf(stderr, "eval requires --load <model file>\n");
        return 2;
    }
    Net net;
    if (!net.load(load_path)) return 1;
    std::printf("loaded model from %s\n", load_path.c_str());

    std::vector<Example> Xte;
    if (!load_split(dir, false, 2007, Xte, 1338)) return 1;
    print_eval("test", eval_split(net, Xte));
    return 0;
}

// Try several learning rates, train a fresh net for each, report the best.
static int run_sweep(const std::string& dir, int epochs,
                     const std::vector<double>& lrs) {
    std::vector<Example> Xtr, Xte;
    if (!load_split(dir, true,  7291, Xtr, 1337)) return 1;
    if (!load_split(dir, false, 2007, Xte, 1338)) return 1;
    std::printf("sweep over %zu learning rates, %d epochs each\n", lrs.size(), epochs);

    double best_lr = 0; double best_err = 1e9;
    std::vector<EvalResult> results;
    for (double lr : lrs) {
        Net net; net.init(1337);              // identical init each run -> fair comparison
        std::printf("  training lr=%-8g ... ", lr); std::fflush(stdout);
        EvalResult r = train_net(net, Xtr, Xte, lr, epochs, /*verbose=*/false);
        results.push_back(r);
        std::printf("test error %.2f%% (misses %d, loss %e)\n", r.err * 100.0, r.misses, r.loss);
        if (r.err < best_err) { best_err = r.err; best_lr = lr; }
    }

    std::printf("\n%-12s %-12s %-10s %s\n", "lr", "test err%", "misses", "test loss");
    std::printf("------------------------------------------------\n");
    for (size_t i = 0; i < lrs.size(); ++i)
        std::printf("%-12g %-12.2f %-10d %e%s\n",
                    lrs[i], results[i].err * 100.0, results[i].misses, results[i].loss,
                    lrs[i] == best_lr ? "  <-- best" : "");
    std::printf("\nbest learning rate: %g (test error %.2f%%)\n", best_lr, best_err * 100.0);
    return 0;
}

// ===========================================================================
static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s gradcheck\n"
        "      verify backprop against numerical gradients (no data needed)\n"
        "  %s train [data_dir] [lr] [--epochs N] [--save FILE]\n"
        "      train the net (default lr 0.03, 23 epochs); optionally save weights\n"
        "  %s eval  [data_dir] --load FILE\n"
        "      load a saved model and report test-set error (inference only)\n"
        "  %s sweep [data_dir] [--epochs N] [lr1 lr2 ...]\n"
        "      train one net per learning rate and report the best\n"
        "      (default lrs: 0.003 0.01 0.03 0.1 0.3, default 10 epochs)\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "gradcheck";

    // collect positional args and --flags from argv[2..]
    std::vector<std::string> pos;
    std::string save_path, load_path;
    int epochs = -1; // sentinel -> use per-mode default
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--save" && i + 1 < argc)       save_path = argv[++i];
        else if (a == "--load" && i + 1 < argc)  load_path = argv[++i];
        else if (a == "--epochs" && i + 1 < argc) epochs = std::atoi(argv[++i]);
        else pos.push_back(a);
    }
    // first positional (if not a number) is the data dir
    std::string dir = "data";
    size_t p = 0;
    if (!pos.empty() && pos[0].find_first_of("0123456789") != 0) { dir = pos[0]; p = 1; }

    if (mode == "gradcheck") {
        return run_gradcheck();
    } else if (mode == "train") {
        double lr = (p < pos.size()) ? std::atof(pos[p].c_str()) : 0.03;
        return run_train(dir, lr, epochs < 0 ? 23 : epochs, save_path);
    } else if (mode == "eval") {
        return run_eval(dir, load_path);
    } else if (mode == "sweep") {
        std::vector<double> lrs;
        for (size_t i = p; i < pos.size(); ++i) lrs.push_back(std::atof(pos[i].c_str()));
        if (lrs.empty()) lrs = {0.003, 0.01, 0.03, 0.1, 0.3};
        return run_sweep(dir, epochs < 0 ? 10 : epochs, lrs);
    } else {
        usage(argv[0]);
        return 2;
    }
}
