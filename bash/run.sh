#!/usr/bin/env bash
# Example batch: several GuideFlow3D runs (appearance + similarity).
# Run from anywhere; paths are resolved to the repository root.
# Set BLENDER_INSTALLATION_PATH or BLENDER_HOME if Blender is not under ~/Downloads.

export PYTHONWARNINGS="ignore"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BLENDER_LINK="${BLENDER_LINK:-https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz}"
BLENDER_INSTALLATION_PATH="${BLENDER_INSTALLATION_PATH:-$HOME/Downloads}"
export BLENDER_HOME="${BLENDER_HOME:-$BLENDER_INSTALLATION_PATH/blender-3.0.1-linux-x64/blender}"

install_blender() {
    if [[ -x "$BLENDER_HOME" ]]; then
        echo "Blender already available at $BLENDER_HOME"
        return 0
    fi
    echo "Installing Blender..."
    sudo apt-get update
    sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6
    wget "$BLENDER_LINK" -P "$BLENDER_INSTALLATION_PATH"
    tar -xvf "${BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz" -C "$BLENDER_INSTALLATION_PATH"
    echo "Blender installed at $BLENDER_HOME"
}

install_blender

# Appearance guidance (rendered appearance from mesh)
python run.py \
  --guidance_mode appearance \
  --appearance_mesh examples/B07QC84LP1.glb \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment1 \
  --convert_yup_to_zup

# Appearance guidance (explicit appearance image)
python run.py \
  --guidance_mode appearance \
  --appearance_mesh examples/B07QC84LP1.glb \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment2 \
  --appearance_image examples/B07QC84LP1_orig.png \
  --convert_yup_to_zup

# Similarity guidance (text prompt)
python run.py \
  --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment3 \
  --appearance_text "A light-colored wooden chair with a straight-back design, cushioned rectangular backrest and seat in light beige, slightly outward back legs, and tapered front legs." \
  --convert_yup_to_zup

# Similarity guidance (reference image)
python run.py \
  --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/experiment4 \
  --appearance_image examples/B07QC84LP1_orig.png \
  --convert_yup_to_zup
