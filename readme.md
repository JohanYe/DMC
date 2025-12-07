# From Mesh Completion to AI Designed Crown

This is a quick attempt to fix the broken unmaintained repo. This is NOT an official implementation.

[Original paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_53)

## Requirements

- Python 3.9
- CUDA 11.x
- CUDA toolkit (nvcc) available for compiling extensions

## Installation

Run the setup script on a machine with GPU and CUDA toolkit:
```bash
./uv.sh
source .venv/bin/activate
```

This installs PyTorch 2.1, dependencies, and builds the CUDA extensions.