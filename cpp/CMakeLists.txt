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

add_executable(lecun1989 lecun1989.cpp)

if(MSVC)
    target_compile_options(lecun1989 PRIVATE /O2)
else()
    target_compile_options(lecun1989 PRIVATE -O2)
endif()
