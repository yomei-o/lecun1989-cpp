// ---------------------------------------------------------------------------
//  predict.cpp
//
//  Load a model trained by `lecun1989 train --save model.bin`, read an ordinary
//  image file of a single handwritten digit (PNG / JPG / BMP / ...) with
//  stb_image, preprocess it into the 16x16 [-1,1] format the 1989 net expects
//  (using stb_image_resize2 for the downscale), run the forward pass and print
//  the predicted digit.
//
//  Build: compile together with stb_impl.cpp (which holds the stb
//  implementations) -- see CMakeLists.txt / README.
//
//  Usage:
//     predict <image> [--model FILE] [--invert|--no-invert] [--no-center]
//
//        <image>        path to the digit image (any format stb_image reads)
//        --model FILE   model file to load          (default: model.bin)
//        --invert       force colour inversion (dark digit on light paper)
//        --no-invert    force NO inversion          (already white-on-black)
//                       (default: auto-detect from the border brightness)
//        --no-center    skip bounding-box crop + centering; just scale the
//                       whole image to 16x16
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "lecun_net.h"
#include "stb_image.h"
#include "stb_image_resize2.h"

// The network wants an "ink" convention: background = 0, digit stroke = 1
// (this maps to the MNIST-style white-on-black the net was trained on).
//
// Given a grayscale image `g` (0..255, w*h), produce a float ink buffer in
// [0,1]. `invert` flips it so a dark-on-light scan becomes bright-on-dark.
static std::vector<float> to_ink(const unsigned char* g, int w, int h, bool invert) {
    std::vector<float> ink((size_t)w * h);
    for (size_t i = 0; i < ink.size(); ++i) {
        float v = g[i] / 255.0f;              // 0..1
        ink[i] = invert ? (1.0f - v) : v;     // ink = bright stroke
    }
    return ink;
}

// Decide whether to invert: MNIST digits are bright on a dark background.
// Photographs/scans of handwriting are usually the opposite (dark ink, light
// paper). We look at the average brightness of the image border: if the border
// is bright, the paper is light and we should invert.
static bool auto_should_invert(const unsigned char* g, int w, int h) {
    double sum = 0; int n = 0;
    for (int x = 0; x < w; ++x) { sum += g[x]; sum += g[(size_t)(h - 1) * w + x]; n += 2; }
    for (int y = 0; y < h; ++y) { sum += g[(size_t)y * w]; sum += g[(size_t)y * w + (w - 1)]; n += 2; }
    double border_mean = sum / n;             // 0..255
    return border_mean > 127.0;               // light border -> invert
}

// Crop the ink to its bounding box (pixels above a fraction of the peak),
// re-center it in a square canvas with a margin, and return that square buffer
// (side = out_side). Mirrors the spirit of MNIST's size-normalise + center.
static std::vector<float> crop_and_center(const std::vector<float>& ink, int w, int h,
                                          int& out_side) {
    float peak = 0.0f;
    for (float v : ink) peak = std::max(peak, v);
    const float thr = peak * 0.30f;           // ink threshold relative to peak

    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (ink[(size_t)y * w + x] > thr) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }

    if (x1 < x0 || y1 < y0) {                 // blank image: fall back to whole
        out_side = std::max(w, h);
        std::vector<float> sq((size_t)out_side * out_side, 0.0f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                sq[(size_t)y * out_side + x] = ink[(size_t)y * w + x];
        return sq;
    }

    int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    int side = std::max(bw, bh);
    int margin = side / 4;                     // ~20% border, MNIST-like
    int canvas = side + 2 * margin;
    out_side = canvas;

    std::vector<float> sq((size_t)canvas * canvas, 0.0f);   // background = 0
    int offx = margin + (side - bw) / 2;       // center the bbox in the square
    int offy = margin + (side - bh) / 2;
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x)
            sq[(size_t)(offy + y) * canvas + (offx + x)] =
                ink[(size_t)(y0 + y) * w + (x0 + x)];
    return sq;
}

int main(int argc, char** argv) {
    std::string image_path, model_path = "model.bin";
    int invert_mode = -1;                       // -1 auto, 0 off, 1 on
    bool center = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)      model_path = argv[++i];
        else if (a == "--invert")                invert_mode = 1;
        else if (a == "--no-invert")             invert_mode = 0;
        else if (a == "--no-center")             center = false;
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        } else if (image_path.empty())           image_path = a;
    }

    if (image_path.empty()) {
        std::fprintf(stderr,
            "usage: predict <image> [--model FILE] [--invert|--no-invert] [--no-center]\n");
        return 2;
    }

    // ---- load model ----
    Net net;
    if (!net.load(model_path)) {
        std::fprintf(stderr, "ERROR: could not load model '%s'.\n"
            "Train one first, e.g.:  lecun1989 train data 0.03 --save %s\n",
            model_path.c_str(), model_path.c_str());
        return 1;
    }

    // ---- load image as grayscale ----
    int w = 0, h = 0, ch = 0;
    unsigned char* g = stbi_load(image_path.c_str(), &w, &h, &ch, 1); // force 1 channel
    if (!g) {
        std::fprintf(stderr, "ERROR: could not read image '%s': %s\n",
                     image_path.c_str(), stbi_failure_reason());
        return 1;
    }
    std::printf("loaded model '%s' and image '%s' (%dx%d, %d channel(s))\n",
                model_path.c_str(), image_path.c_str(), w, h, ch);

    bool invert = (invert_mode == -1) ? auto_should_invert(g, w, h)
                                      : (invert_mode == 1);
    std::printf("colour inversion: %s%s\n", invert ? "on" : "off",
                invert_mode == -1 ? " (auto)" : "");

    // ---- preprocess to a 16x16 ink field, then to [-1,1] ----
    std::vector<float> ink = to_ink(g, w, h, invert);
    stbi_image_free(g);

    int side = 0;
    std::vector<float> sq = center ? crop_and_center(ink, w, h, side)
                                   : (side = std::max(w, h), ink);
    int sw = center ? side : w;
    int sh = center ? side : h;

    float small16[dim::IN * dim::IN];
    stbir_resize_float_linear(sq.data(), sw, sh, 0,
                              small16, dim::IN, dim::IN, 0,
                              STBIR_1CHANNEL);

    double x[dim::IN * dim::IN];
    for (int i = 0; i < dim::IN * dim::IN; ++i)
        x[i] = 2.0 * (double)small16[i] - 1.0;   // ink [0,1] -> [-1,1]

    // ---- forward + report ----
    net.forward(x);
    int pred = net.argmax_yhat();

    std::printf("\n  predicted digit: %d\n\n", pred);
    std::printf("  per-class scores (tanh output, higher = more likely):\n");

    // rank classes by score for a readable summary
    int order[dim::OUT];
    for (int i = 0; i < dim::OUT; ++i) order[i] = i;
    std::sort(order, order + dim::OUT,
              [&](int a, int b) { return net.yhat[a] > net.yhat[b]; });
    for (int r = 0; r < dim::OUT; ++r) {
        int d = order[r];
        // map tanh output [-1,1] to a 0..40 bar for a quick visual
        int bar = (int)std::lround((net.yhat[d] + 1.0) * 0.5 * 40.0);
        bar = std::max(0, std::min(40, bar));
        std::printf("    %d: %+.4f  %s%s\n", d, net.yhat[d],
                    std::string(bar, '#').c_str(),
                    d == pred ? "  <-- prediction" : "");
    }
    return 0;
}
