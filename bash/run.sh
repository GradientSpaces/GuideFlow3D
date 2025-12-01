export PYTHONWARNINGS="ignore"


# Blender installation configuration
BLENDER_LINK='https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH='/tmp'
export BLENDER_HOME="${BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"

# Function to install Blender
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

# Appearance Guidance (with rendered image)
python run.py \
  --guidance_mode appearance \
  --appearance_mesh examples/B07QC84LP1.glb \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment1 \
  --convert_yup_to_zup \

# # Appearance Guidance
python run.py \
  --guidance_mode appearance \
  --appearance_mesh examples/B07QC84LP1.glb \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment2 \
  --appearance_image examples/B07QC84LP1_orig.png \
  --convert_yup_to_zup

# Similarity Guidance (with text prompt)
python run.py \
  --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment3 \
  --appearance_text "A light-colored wooden chair with a straight-back design, cushioned rectangular backrest and seat in light beige, slightly outward back legs, and tapered front legs." \
  --convert_yup_to_zup

# Similarity Guidance (with reference image)
python run.py \
  --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment4 \
  --appearance_image examples/B07QC84LP1_orig.png \
  --convert_yup_to_zup