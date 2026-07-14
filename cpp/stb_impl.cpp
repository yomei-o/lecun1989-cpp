// Single translation unit that compiles the stb implementations.
//
// Keeping the implementations here (rather than in predict.cpp) avoids the
// implementation-only macro redefinitions inside stb_image_resize2.h from
// clobbering the public STBIR_* pixel-layout enum names that predict.cpp uses.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
