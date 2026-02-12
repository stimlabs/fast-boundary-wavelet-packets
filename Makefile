# fast-boundary-wavelet-packets — project Makefile
# ------------------------------------------------
# Configuration — override on the command line or via environment variables.
TORCH_DIR     ?= $(error Set TORCH_DIR to <libtorch>/share/cmake/Torch, e.g. make build TORCH_DIR=/opt/libtorch/share/cmake/Torch)
BUILD_DIR     := build
BUILD_TYPE    ?= Release
REFERENCE_DIR := reference
REFERENCE_URL := https://github.com/v0lta/PyTorch-Wavelet-Toolbox.git

# ------------------------------------------------
# Phony targets
.PHONY: all setup setup-reference setup-python build test clean

all: setup build

# ---------- Setup ---------------------------------------------------

## Clone reference implementation + install Python deps.
setup: setup-reference setup-python

## Clone the reference Python library into reference/.
setup-reference:
	@if [ ! -d "$(REFERENCE_DIR)/.git" ]; then \
		echo "Cloning reference implementation …"; \
		git clone --depth 1 $(REFERENCE_URL) $(REFERENCE_DIR); \
	else \
		echo "Reference repo already present — pulling latest …"; \
		git -C $(REFERENCE_DIR) pull --ff-only; \
	fi

## Install Python dependencies into the active virtual environment.
## Fails if no virtual environment is active or discoverable.
setup-python:
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "Using active virtualenv: $$VIRTUAL_ENV"; \
	elif [ -d ".venv" ]; then \
		echo "Activating .venv …"; \
		. .venv/bin/activate; \
	else \
		echo "ERROR: No virtual environment found." >&2; \
		echo "" >&2; \
		echo "  Create one first, e.g.:" >&2; \
		echo "    python -m venv .venv && source .venv/bin/activate" >&2; \
		echo "" >&2; \
		echo "  Or activate an existing one before running make." >&2; \
		exit 1; \
	fi; \
	pip install -r requirements.txt

# ---------- Build ---------------------------------------------------

## Configure and build the C++ project with CMake.
build:
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DTorch_DIR=$(TORCH_DIR)
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) -j

# ---------- Test ----------------------------------------------------

## Run C++ tests via CTest.
test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# ---------- Clean ---------------------------------------------------

## Remove build artifacts.
clean:
	rm -rf $(BUILD_DIR)
