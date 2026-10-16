Third Party Software
====================

This code uses the following third party software/source code.

## Premake

- Where: https://premake.github.io/download
- What: Build configuration and generator
- License: BSD 3-Clause

Premake generates platform/toolchain specific build files, such as Visual
Studio project files or standard Makefiles. This project includes prebuilt
binaries for Linux and Windows.

## Snitch

- Where: https://github.com/snitch-org/snitch
- What: Lightweight C++20 testing framework.
- License: Boost Software License 1.0

Snitch is a C++20 testing framework.

Slightly modified to fix some problems with recent MSVC and Apple/Clang. See
lines with // **MODIFIED**

## nanobench

- Where: https://nanobench.ankerl.com/ and https://github.com/martinus/nanobench
- What: "Simple, fast, accurate single-header microbenchmarking [...]"
- License: MIT

Nanobench is a small platform-independent library for microbenchmarking. It 
also has a funny cute logo.
