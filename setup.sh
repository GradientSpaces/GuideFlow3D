conda create -n guideflow3d python=3.10 -y
conda activate guideflow3d
conda init

conda install pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.4 torchaudio==2.5.0 -c pytorch -c nvidia

# basic
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# xformers
pip install xformers==0.0.28.post2 --no-deps --index-url https://download.pytorch.org/whl/cu124

# flash-attn
pip install flash-attn

# # nvdiffrast
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast

# # # diffoctreerast
mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install /tmp/extensions/diffoctreerast

# # kaolin
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# mipgaussian
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# spconv
pip install spconv-cu120 

# Partfield
conda install nvidia/label/cuda-12.4.0::cuda -y
pip install psutil
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
sudo apt install libx11-6 libgl1 libxrender1
pip install vtk

# python-pycg
pip install -U 'python-pycg[all]'

# numpy version issue
pip install tetgen==0.6.4
pip install numpy==1.26.4