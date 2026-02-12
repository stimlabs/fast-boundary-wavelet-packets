# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

C++ (LibTorch) implementation of the wavelet packet transform with special boundary filters, supporting QR and Gram-Schmidt orthogonalization. Based on the Python [PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox) reference implementation (cloned into `reference/`).

This is a focused reimplementation — only the boundary-filter wavelet packet transform is in scope.

## Build and test commands

Requires `TORCH_DIR` pointing to `<libtorch>/share/cmake/Torch`:

```bash
# Full build
make build TORCH_DIR=~/libs/libtorch/share/cmake/Torch

# Run all tests
make test

# Run a single CTest by name
ctest --test-dir build --output-on-failure -R <test_name>

# Clean rebuild
make clean && make build TORCH_DIR=~/libs/libtorch/share/cmake/Torch
```

The `TORCH_DIR` can also be set as an environment variable. Build type defaults to Release; override with `BUILD_TYPE=Debug`.

### Python setup (for reference comparison scripts)

```bash
python -m venv .venv && source .venv/bin/activate
make setup          # clones reference repo + pip install
```

`make setup-python` requires an active virtualenv (`VIRTUAL_ENV` env var or `.venv/` directory).

## Project layout

```
CMakeLists.txt         # C++ build — C++17, links LibTorch
Makefile               # Task runner (setup, build, test, clean)
include/               # Public C++ headers
src/                   # C++ implementation
tests/                 # C++ tests
scripts/               # Python helper/comparison scripts
reference/             # Cloned PyTorch-Wavelet-Toolbox (gitignored)
```

## Code style

- **East const**: `const` always goes after the type. Write `int const x`, `std::string const& name`, `char const*`, never `const int x` or `const std::string&`.

## Architecture

### Core mathematical model

The transform is a linear operator: analysis `c = A x`, synthesis `x = S c`, with the requirement `S * A = I` (perfect reconstruction). Interior rows of `A` use standard wavelet filters; boundary rows use short, specially constructed filters to avoid edge artifacts while preserving signal length.

### Orthogonalization modes

Used only for constructing valid boundary filter rows:

- `"qr"`: Dense QR factorization. More numerically robust, higher memory.
- `"gramschmidt"`: Sparse/in-place Gram-Schmidt. More memory efficient, potentially less stable.

### Key design differences from reference

- The wavelet packet tree is **not built lazily** — all levels up to `max_level` are computed in one pass.
- Subbands at each level are tiled into a structured representation with the same spatial extent as the input, enabling stacking along a scale dimension.

## Success criteria

- Forward/inverse transforms satisfy `S * A ~ I` within numerical tolerance.
- Coefficients match the reference Python implementation for both `float32` and `float64`.
- Both orthogonalization modes produce consistent results.
