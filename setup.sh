conda create -n guideflow3d python=3.11 -y
conda activate guideflow3d
conda init

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# basic
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# xformers
pip install xformers==0.0.31 --no-deps --index-url https://download.pytorch.org/whl/cu128

# flash-attn
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install --no-build-isolation -e .
cd ..
rm -rf flash-attention

# # nvdiffrast
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-build-isolation /tmp/extensions/nvdiffrast

# # # diffoctreerast
mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install --no-build-isolation /tmp/extensions/diffoctreerast

# # kaolin
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html

# mipgaussian
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# spconv
pip install spconv-cu126 

# Partfield
conda install nvidia/label/cuda-12.8.0::cuda -y
pip install psutil
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
sudo apt install libx11-6 libgl1 libxrender1
pip install vtk

# python-pycg
pip install -U 'python-pycg[all]'

# Version Issues
pip install tetgen==0.6.4
pip install numpy==1.26.4
pip install opencv-python-headless==4.11.0.86
pip install opencv-python==4.11.0.86
pip install rembg==2.0.41
