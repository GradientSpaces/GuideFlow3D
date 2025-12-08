#!/bin/bash

# Create a directory to store wheels
mkdir -p ./wheels

# Update system packages
apt-get update -y
apt-get install -y xvfb libx11-6 libgl1 libxrender1

# 1. Basic Dependencies
# We use 'pip wheel' to build/download wheels instead of installing
pip wheel --wheel-dir=./wheels \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip wheel --wheel-dir=./wheels \
    pyvirtualdisplay \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista \
    pymeshfix igraph transformers tensorview psutil \
    lightning==2.2 h5py yacs scikit-image loguru boto3 \
    mesh2sdf tetgen==0.6.4 pymeshlab plyfile einops libigl \
    polyscope potpourri3d simple_parsing arrgh vtk numpy==1.26.4

# 2. Git Repositories
# pip wheel handles git urls perfectly
pip wheel --wheel-dir=./wheels \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 3. Extensions with Custom Build Steps (nvdiffrast, diffoctreerast, mip-splatting)
# These often require cloning first if they have submodules or complex setups

# nvdiffrast
mkdir -p /tmp/extensions
if [ ! -d "/tmp/extensions/nvdiffrast" ]; then
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
pip wheel --wheel-dir=./wheels /tmp/extensions/nvdiffrast

# diffoctreerast
if [ ! -d "/tmp/extensions/diffoctreerast" ]; then
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
fi
pip wheel --wheel-dir=./wheels /tmp/extensions/diffoctreerast

# mip-splatting (diff-gaussian-rasterization)
if [ ! -d "/tmp/extensions/mip-splatting" ]; then
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
fi
pip wheel --wheel-dir=./wheels /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# 4. Pre-built Wheels (Kaolin, torch-scatter, spconv)
# These are already wheels, so we just download them to the folder
pip download --dest ./wheels \
    kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

pip download --dest ./wheels \
    spconv-cu124

pip download --dest ./wheels \
    torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# 5. Python-PyCG
pip wheel --wheel-dir=./wheels 'python-pycg[all]'

echo "All wheels built in ./wheels"
