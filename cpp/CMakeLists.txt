# Cross-platform build for the 1989 LeCun ConvNet C++ port.
# Works with MSVC (Visual Studio 2022), GCC and Clang on any OS.
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
cmake_minimum_required(VERSION 3.15)
project(lecun1989 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimised build if the user did not pick one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Trainer / evaluator (self-contained, no third-party deps).
add_executable(lecun1989 lecun1989.cpp)

# Image classifier: loads a saved model and predicts the digit in an image.
# stb_image / stb_image_resize2 implementations live in stb_impl.cpp.
add_executable(predict predict.cpp stb_impl.cpp)

foreach(tgt lecun1989 predict)
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /O2)
    else()
        target_compile_options(${tgt} PRIVATE -O2)
    endif()
endforeach()
