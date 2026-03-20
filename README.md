<p align="center">
  <h2 align="center"> GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer </h2>
  <p align="center">
    <a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup> 1 </sup>
    .
    <a href="https://vevenom.github.io/">Sinisa Stekovic</a><sup> 2 </sup>
    .
    <a href="https://vincentlepetit.github.io/">Vincent Lepetit</a><sup> 2 </sup>
    .
    <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>
  </p>
  <p align="center"> <strong>Neural Information Processing Systems (NeurIPS) 2025</strong></p>
  <p align="center">
    <sup> 1 </sup>Stanford University · <sup> 2 </sup>ENPC, IP Paris
  </p>
  <h3 align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2510.16136)
 [![ProjectPage](https://img.shields.io/badge/Project_Page-GuideFlow3D-blue)](https://sayands.github.io/guideflow3d)
 [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
 <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/sayands/guideflow3d/blob/main/assets/guideflow3d_teaser.gif" width="100%">
  </a>
</p>

<h5 align="left">
<em>TL;DR:</em> 3D appearance transfer pipeline robust to strong geometric variations between objects.
</h5>

## 📃 Abstract

Transferring appearance to 3D assets using different representations of the appearance object—such as images or text—has garnered interest due to its wide range of applications in industries like gaming, augmented reality, and digital content creation. However, state-of-the-art methods still fail when the geometry between the input and appearance objects is significantly different. A straightforward approach is to directly apply a 3D generative model, but we show that this ultimately fails to produce appealing results. Instead, we propose a principled approach inspired by universal guidance. Given a pretrained rectified flow model conditioned on image or text, our training-free method interacts with the sampling process by periodically adding guidance. This guidance can be modeled as a differentiable loss function, and we experiment with two different types of guidance including part-aware losses for appearance and self-similarity. Our experiments show that our approach successfully transfers texture and geometric details to the input 3D asset, outperforming baselines both qualitatively and quantitatively. We also show that traditional metrics are not suitable for evaluating the task due to their inability of focusing on local details and comparing dissimilar inputs, in absence of ground truth data. We thus evaluate appearance transfer quality with a GPT-based system objectively ranking outputs, ensuring robust and human-like assessment, as further confirmed by our user study. Beyond showcased scenarios, our method is general and could be extended to different types of diffusion models and guidance functions.

_**Check out our [Project Page](https://sayands.github.io/guideflow3d) for more examples and interactive demos!**_

## 📰 News

- ![](https://img.shields.io/badge/New!-8A2BE2) **[2026-03]** Reference **code** and interactive **Viser** demo released — see **Installation** & **Usage** below.
- **[2025-09]** 🎉🥳 GuideFlow3D **accepted** to **NeurIPS 2025**! See you in San Diego 🔥✨

## 📦 Installation

Tested on **Ubuntu 22.04.05 LTS**, **CUDA 12.8**, **PyTorch 2.7.1**.

**1. Clone the repo and install conda.**
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda so `conda` is on your `PATH`. Linux x86_64 one-liner (silent install to `~/miniconda3`, then hook your shell):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
bash miniconda.sh -b -p "$HOME/miniconda3" 
rm miniconda.sh
"$HOME/miniconda3/bin/conda" init
```

**2. Create and activate the conda environment.**

```bash
conda create -n guideflow3d python=3.11 -y
conda activate guideflow3d
```

**3. Run `setup.sh`.**
From the repo root, run `bash setup.sh`. This installs PyTorch and all dependencies, and builds compiled packages (flash-attention, nvdiffrast, etc.). The install is slow; `sudo` may be needed for system packages.

**4. Install Blender 3.0.1** (Linux x64).
Multiview rendering uses the TRELLIS Blender script. Download and extract the [3.0.1 tarball](https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz), then point `BLENDER_HOME` at the binary:

```bash
export BLENDER_INSTALLATION_PATH="$HOME/Downloads"
export BLENDER_HOME="$BLENDER_INSTALLATION_PATH/blender-3.0.1-linux-x64/blender"
```

Add these to your `.bashrc`/`.zshrc` or export them in the same session before running the pipeline.

**5. Download PartField weights.**
Download the **Objaverse** pretrained model from [PartField](https://github.com/nv-tlabs/PartField) (*Pretrained Model* in their README) and place `model_objaverse.ckpt` in the `weights/` directory at the repository root (gitignored). The path must match [`continue_ckpt`](third_party/PartField/config.yaml) in `third_party/PartField/config.yaml`.

<details>
<summary><b>Troubleshooting</b></summary>

- If `conda activate` does not apply after `setup.sh`, open a new terminal and run `conda activate guideflow3d`.
- If Blender is missing at runtime, verify `BLENDER_HOME` points to the extracted `blender-3.0.1-linux-x64/blender` binary. Set `BLENDER_INSTALLATION_PATH` if you did not use `~/Downloads`.
- If PartField inference fails, confirm `weights/model_objaverse.ckpt` exists and matches `third_party/PartField/config.yaml`.

</details>

## 🗂️ Project Structure

```
guideflow3d/
├── run.py                  # main CLI entry point
├── config/default.yaml     # default hyperparameters
├── gui/app.py              # Viser interactive demo
├── bash/run.sh             # example batch runs
├── lib/
│   ├── opt/                # guidance optimization
│   │   ├── appearance.py   #   appearance (part-aware) guidance
│   │   └── self_similarity.py  #   self-similarity guidance
│   └── util/               # pipeline utilities
│       ├── generation.py   #   DINOv2 feature extraction, SLAT encoding/decoding
│       ├── render.py       #   Blender multiview rendering
│       ├── pointcloud.py   #   mesh voxelization
│       └── partfield.py    #   PartField feature sampling & co-segmentation
├── third_party/            # vendored TRELLIS and PartField
├── examples/               # sample meshes and images
└── weights/                # PartField checkpoint (gitignored)
```

## 💡 Usage

Work from the **repository root** with the `guideflow3d` env active. Steps 4–5 of [Installation](#-installation) (Blender + PartField weights) are required before running anything below. We provide sample meshes and images under [`examples/`](examples/).

### Demo

To start the web-based interactive demo:

```bash
python gui/app.py
```

Open **http://localhost:8080** in your browser ([Viser](https://github.com/nerfstudio-project/viser)). Outputs default to `outputs/gui_run_<id>/`; use **Toggle Structure / Output** to compare the output mesh against the input structure mesh.

### `run.py` (main script)

[`run.py`](run.py) is the main command-line entry point. It runs the full pipeline end-to-end: Blender rendering → voxelization → PartField co-segmentation → (appearance only) DINOv2 feature extraction → SLAT encoding → guided flow sampling → SLAT decoding.

| Argument | Required | Description |
|----------|----------|-------------|
| `--guidance_mode` | Yes | `appearance` or `similarity` |
| `--structure_mesh` | Yes | Structure mesh (`.glb`) |
| `--output_dir` | Yes | Outputs, renders, checkpoints |
| `--convert_yup_to_zup` | No | Y-up → Z-up |
| `--appearance_mesh` | Appearance | Appearance mesh (`.glb`) |
| `--appearance_image` | No* | Reference image |
| `--appearance_text` | No* | Text (similarity mode) |

\* **Similarity:** `--appearance_text` **or** `--appearance_image`, not both. **Appearance:** `--appearance_mesh` required; without `--appearance_image`, an image is rendered from the mesh.

```bash
python run.py --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/my_run \
  --appearance_text "a wooden chair"
```

**Outputs.** Each run writes the following under `--output_dir`:

| File | Description |
|------|-------------|
| `out_app.glb` / `out_sim.glb` | Final textured mesh (appearance / similarity mode) |
| `out_gaussian_app.mp4` / `out_gaussian_sim.mp4` | Gaussian splatting turntable video |
| `struct_renders/`, `app_renders/` | Blender multiview renders |
| `voxels/` | Voxelized structure (and appearance) meshes |
| `features/`, `latents/` | DINOv2 features and SLAT latents (appearance mode) |
| `partfield/` | PartField feature planes |

**Expected runtime:** ~2–4 minutes on a single RTX 4090 (appearance mode, 300 steps). Similarity mode is faster since it skips DINOv2 extraction and SLAT encoding.

### `bash/run.sh` (a few examples)

[`bash/run.sh`](bash/run.sh) shows a handful of example commands, running several `run.py` jobs on files in [`examples/`](examples/).

```bash
bash bash/run.sh
```

## 🙏 Acknowledgments

- 🧊 **[TRELLIS](https://github.com/microsoft/TRELLIS)** — structured 3D latents, encoders, rendering.
- 🔧 **[PartField](https://github.com/nv-tlabs/PartField)** — part-aware 3D feature fields.
- 🎛️ **[SpaceControl](https://github.com/spacecontrol3d/spacecontrol)** — Viser GUI ideas.

## 📜 Citation

```bibtex
@inproceedings{sdsarkar_guideflow3d_2025,
      author = {Deb Sarkar, Sayan and Stekovic, Sinisa and Lepetit, Vincent and Armeni, Iro},
      title = {GuideFlow3D: Optimization-Guided Rectified Flow For 3D Appearance Transfer},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```
