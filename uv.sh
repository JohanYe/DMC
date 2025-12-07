#!/bin/bash
set -euo pipefail

uv venv .venv --python 3.9
source .venv/bin/activate

# PyTorch 2.1 + CUDA 11.8
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
uv pip install \
    easydict \
    h5py \
    matplotlib \
    numpy \
    opencv-python \
    pyyaml \
    scipy \
    tensorboardX \
    tensorboard \
    "timm==0.4.5" \
    tqdm \
    transforms3d \
    trimesh \
    plyfile \
    scikit-image \
    python-mnist \
    av \
    pykdtree \
    ipdb \
    libigl \
    ninja

# open3d - 0.9 won't work, use a compatible version for py3.9
uv pip install "open3d>=0.17,<0.18"

# KNN CUDA
uv pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# torch-scatter for torch 2.1 + cu118
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# pointnet2_ops (compiles CUDA)
uv pip install git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops\&subdirectory=pointnet2_ops_lib --no-build-isolation
# pytorch3d prebuilt for torch 2.1 + cu118 + py3.9
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Build CUDA extensions
PROJ_ROOT=$(pwd)
for ext in chamfer_dist cubic_feature_sampling gridding gridding_loss; do
    cd "$PROJ_ROOT/extensions/$ext"
    python setup.py install
done
cd "$PROJ_ROOT"