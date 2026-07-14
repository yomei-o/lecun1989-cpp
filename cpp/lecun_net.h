// ---------------------------------------------------------------------------
//  lecun_net.h
//
//  The 1989 LeCun ConvNet itself: dimensions, sparse connectivity, the Net
//  struct (parameters, forward/backward, SGD, and model save/load).
//
//  Shared by both the trainer (lecun1989.cpp) and the image classifier
//  (predict.cpp) so there is a single source of truth for the architecture
//  and the on-disk model format.
//
//  Self-contained standard C++17 -- no third party libraries.
// ---------------------------------------------------------------------------
#ifndef LECUN_NET_H
#define LECUN_NET_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <algorithm>

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

#endif // LECUN_NET_H
