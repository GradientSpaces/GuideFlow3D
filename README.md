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

1. **Conda:** install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda so `conda` is on your `PATH`. Linux x86_64 (silent install to `~/miniconda3`, then hook your shell):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p "$HOME/miniconda3" && rm miniconda.sh && "$HOME/miniconda3/bin/conda" init
```

2. From the repo root: `bash setup.sh`

**What [`setup.sh`](setup.sh) does:** creates conda env `guideflow3d` (Python 3.11), installs PyTorch and dependencies, and builds or installs compiled packages (e.g. flash-attention, nvdiffrast). It does **not** install Blender or PartField weights — add those under **Setup** below. The install is slow; **`sudo`** may be used for system packages.

### Setup

Extra assets required to run the full pipeline (not installed by `setup.sh`):

**Blender 3.0.1** (Linux x64) — multiview rendering uses the TRELLIS Blender script. Set **`BLENDER_HOME`** to the `blender` executable. If you extract [the 3.0.1 tarball](https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz) under `~/Downloads`:

```bash
export BLENDER_INSTALLATION_PATH="$HOME/Downloads"   # default in `lib/util/render.py`
export BLENDER_HOME="$BLENDER_INSTALLATION_PATH/blender-3.0.1-linux-x64/blender"
```

Add these to your shell or the same session before Python. If the binary is missing, `lib/util/render.py` may download and extract once (needs **`sudo`** and apt deps like [`bash/run.sh`](bash/run.sh)), or run that script’s `install_blender` block once.

**PartField (Objaverse checkpoint)** — `run.py` loads weights from **`weights/model_objaverse.ckpt`** (see [`continue_ckpt`](third_party/PartField/config.yaml) in `third_party/PartField/config.yaml`). Download the **Objaverse** pretrained model from [PartField](https://github.com/nv-tlabs/PartField) (*Pretrained Model* in their README), place **`model_objaverse.ckpt`** in **`weights/`** at the **repository root** (that directory is gitignored). Licensing and download links are described upstream.

**Troubleshooting:** If **`conda activate`** does not apply after `setup.sh`, open a new terminal and run `conda activate guideflow3d`. If Blender is missing, point **`BLENDER_HOME`** at the extracted `blender-3.0.1-linux-x64/blender` binary; set **`BLENDER_INSTALLATION_PATH`** if you did not use `~/Downloads`. If PartField inference fails, confirm **`weights/model_objaverse.ckpt`** exists and matches **`third_party/PartField/config.yaml`**.

## 💡 Usage

Work from the **repository root**. We provide sample meshes and images under **[`examples/`](examples/)**.

**Blender:** set **`BLENDER_HOME`** as in **Installation → Setup**.

**PartField:** place the Objaverse checkpoint at **`weights/model_objaverse.ckpt`** as in **Installation → Setup** and [PartField](https://github.com/nv-tlabs/PartField). Required for PartField inside **`run.py`** and the GUI.

### Demo → `python gui/app.py`

To start the web-based interactive demo:

```bash
python gui/app.py
```

Open **http://localhost:8080** in your browser ([Viser](https://github.com/nerfstudio-project/viser)). Outputs default to `outputs/gui_run_<id>/`; use **Toggle Structure / Output** to compare output mesh and input structure mesh.

### `run.py` (main script)

[`run.py`](run.py) is the main command-line entry point for the full pipeline. It writes `out_app.glb` or `out_sim.glb` under `--output_dir`, depending on the mode.

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

### `bash/run.sh` (a few examples)

[`bash/run.sh`](bash/run.sh) shows a handful of example commands: on Linux it can install Blender if missing, then runs several `run.py` jobs on files in [`examples/`](examples/).

```bash
bash bash/run.sh
```

## 🙏 Acknowledgments

- 🧊 **[TRELLIS](https://github.com/microsoft/TRELLIS)** — structured 3D latents, encoders, rendering.
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
