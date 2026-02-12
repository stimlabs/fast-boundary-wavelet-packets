# fast-boundary-wavelet-packets

### Project in one sentence

C++ (LibTorch) implementation of the wavelet packet transform with special boundary filters, supporting QR and Gram–Schmidt orthogonalization.

### Reference implementation

Based on the Python PyTorch library [PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox) that implements the wavelet packet transform with special boundary filters and orthogonalization.

### Purpose and scope

Provide a focused, high-performance implementation of the wavelet packet transform for finite inputs that uses special boundary filters and preserves perfect reconstruction.
The code is implemented in C++ using LibTorch and exposes two orthogonalization methods: `"qr"` (dense QR) and `"gramschmidt"` (sparse/in-place Gram–Schmidt).

This is a focused implementation. Other features of the reference library are intentionally not included.

### Makefile targets

| Target                 | Description                                              |
|------------------------|----------------------------------------------------------|
| `make setup`           | Clone reference repo and install Python requirements     |
| `make setup-reference` | Clone/update the reference repo into `reference/`        |
| `make setup-python`    | `pip install -r requirements.txt` (requires active venv) |
| `make build`           | CMake configure + build (requires `TORCH_DIR`)           |
| `make test`            | Run C++ tests via CTest                                  |
| `make clean`           | Remove the `build/` directory                            |

`make setup-python` requires an active virtual environment. It checks for the `VIRTUAL_ENV` environment variable or a local `.venv/` directory. If neither is found, it exits with an error and instructions to create one.

### Project layout

```
.
├── CLAUDE.md
├── CMakeLists.txt         # C++ build configuration
├── Makefile               # Top-level task runner
├── requirements.txt       # Python dependencies
├── include/               # Public C++ headers
├── src/                   # C++ implementation
├── tests/                 # C++ tests
├── scripts/               # Python scripts
├── reference/             # Cloned reference repo
└── build/                 # CMake build directory
```

## Implementation Details

### Core mathematical model

The transform is treated as a linear operator:

* Analysis: `c = A x`
* Synthesis: `x = S c`
* Requirement: `S · A ≈ I` (perfect reconstruction)

Interior rows of `A` use standard wavelet filters.
Boundary rows are replaced by short filters whose coefficients are computed so that the full operator still satisfies the reconstruction and orthogonality constraints.

This becomes a **matrix orthogonalization problem** at the boundaries.

### Orthogonalization modes

Used only for constructing valid boundary filter rows.

* `"qr"`: Dense QR factorization. More numerically robust, higher memory use.
* `"gramschmidt"`: Sparse / in-place Gram–Schmidt. More memory efficient, potentially less stable.

Both must produce operators satisfying reconstruction constraints within numerical tolerance.

## Success criteria

* Forward and inverse transforms must satisfy `S · A ≈ I` within acceptable numerical tolerance.
* Computed coefficients at each level must **match the reference Python implementation** within configurable tolerance for both `float32` and `float64`.
* Both orthogonalization modes (`"qr"` and `"gramschmidt"`) must produce outputs consistent with the above.
