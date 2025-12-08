apt-get update -y
apt-get install -y xvfb
pip install pyvirtualdisplay

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers tensorview -qq
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 -qq
pip install flash-attn

mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast

mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install /tmp/extensions/diffoctreerast

# pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html # CHECK CUDA VERSION BEFORE INSTALLING
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

pip install spconv-cu124

pip install -U 'python-pycg[all]'
pip install psutil
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
sudo apt install libx11-6 libgl1 libxrender1
pip install vtk

pip install tetgen==0.6.4
pip install numpy==1.26.4

mkdir -p ./models
wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt -O ../models/model_objaverse.ckpt

export BLENDER_LINK='https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
export BLENDER_INSTALLATION_PATH='/tmp'
export BLENDER_HOME="${BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"

install_blender() {
    if [ ! -f "$BLENDER_HOME" ]; then
        echo "Installing Blender..."
        sudo apt-get update
        sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6
        wget "$BLENDER_LINK" -P "$BLENDER_INSTALLATION_PATH"
        tar -xvf "${BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz" -C "$BLENDER_INSTALLATION_PATH"
        echo "Blender installed at $BLENDER_HOME"
    else
        echo "Blender already installed at $BLENDER_HOME"
    fi
}

install_blender