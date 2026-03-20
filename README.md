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

- **[2025-09]** GuideFlow3D accepted to **NeurIPS 2025** — see you in San Diego.

## 🚧 Code & data

Code and data are staged for public release; this repository hosts the reference implementation and examples.

## 📦 Installation

Tested on **Ubuntu 22.04** with **CUDA 12.8** and **PyTorch 2.7.1**. Other stacks may work but are not verified.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2. From the **repository root**, run:

```bash
bash setup.sh
```

[`setup.sh`](setup.sh) creates the `guideflow3d` conda env (Python 3.11), installs PyTorch, and pulls in the rest of the stack (including builds such as flash-attention and nvdiffrast). Expect a long install; `sudo` may be needed for system packages.

If `conda activate` does not stick after the script, open a new shell and run `conda activate guideflow3d` before calling `run.py` or the GUI.

## 💡 Usage

Work from the **repository root** (where `run.py` and `config/` live).

### Command line — `run.py`

| Argument | Required | Description |
|----------|----------|-------------|
| `--guidance_mode` | Yes | `appearance` or `similarity` |
| `--structure_mesh` | Yes | Structure mesh (`.glb`) |
| `--output_dir` | Yes | Outputs, renders, checkpoints |
| `--convert_yup_to_zup` | No | Y-up → Z-up |
| `--appearance_mesh` | Appearance | Appearance mesh (`.glb`) |
| `--appearance_image` | No* | Reference image |
| `--appearance_text` | No* | Text (similarity mode) |

\* **Similarity:** use **either** `--appearance_text` **or** `--appearance_image`, not both. **Appearance:** `--appearance_mesh` is required; without `--appearance_image`, an image is rendered from the mesh.

```bash
python run.py --guidance_mode similarity \
  --structure_mesh examples/example1.glb \
  --output_dir outputs/my_run \
  --appearance_text "a wooden chair"
```

Artifacts: `out_app.glb` or `out_sim.glb` under `--output_dir` (mode-dependent).

### Batch examples — `bash/run.sh`

[`bash/run.sh`](bash/run.sh) can install Blender on Linux if needed, then runs several example jobs from the repo root:

```bash
bash bash/run.sh
```

Optional env overrides:

```bash
export BLENDER_INSTALLATION_PATH="$HOME/Downloads"
export BLENDER_HOME="/path/to/blender-3.0.1-linux-x64/blender"
bash bash/run.sh
```

### Interactive GUI — `gui/app.py`

[Viser](https://github.com/nerfstudio-project/viser)-based UI around `run.py` (layout inspired by [SpaceControl](https://github.com/spacecontrol3d/spacecontrol)): load mesh, choose mode, upload / prompt, generate with live logs.

```bash
python gui/app.py
```

Open **http://localhost:8080**. Runs write under `outputs/gui_run_<id>/` by default; use **Toggle Structure / Output** to compare with `out_app.glb` / `out_sim.glb`.

## 🙏 Acknowledgments

- 🧊 We thank the authors of **[TRELLIS](https://github.com/microsoft/TRELLIS)** for structured 3D latents, encoders, and rendering code used in this pipeline.
- 🎛️ The interactive GUI builds on ideas from **[SpaceControl](https://github.com/spacecontrol3d/spacecontrol)** — see also the **[project page](https://spacecontrol3d.github.io/)**.

## 💬 Contact

Questions: GitHub **Issues** or Sayan Deb Sarkar (**sdsarkar@stanford.edu**).

## 📜 Citation

```bibtex
@inproceedings{sayandsarkar_2025_guideflow3d,
      author = {Deb Sarkar, Sayan and Stekovic, Sinisa and Lepetit, Vincent and Armeni, Iro},
      title = {GuideFlow3D: Optimization-Guided Rectified Flow For 3D Appearance Transfer},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```
