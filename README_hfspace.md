---
title: GuideFlow3D
emoji: 🤗
colorFrom: yellow
colorTo: blue
app_file: demos/run_gradio_demo.py
sdk: gradio
sdk_version: 6.0.1
pinned: false
license: apache-2.0
python_version: "3.12"
short_description: A HF Space that demonstrates all use-cases for GuideFlow3D
---

Note: If you're looking to get a working version of the Hugging Face Space, then please use [this link](https://huggingface.co/spaces/suvadityamuk/GuideFlow3D) to fetch files with all assets in place as well.

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

Transferring appearance to 3D assets using different representations of the appearance object - such as images or text - has garnered interest due to its wide range of applications in industries like gaming, augmented reality, and digital content creation. However, state-of-the-art methods still fail when the geometry between the input and appearance objects is significantly different. A straightforward approach is to directly apply a 3D generative model, but we show that this ultimately fails to produce appealing results. Instead, we propose a principled approach inspired by universal guidance. Given a pretrained rectified flow model conditioned on image or text, our training-free method interacts with the sampling process by periodically adding guidance. This guidance can be modeled as a differentiable loss function, and we experiment with two different types of guidance including part-aware losses for appearance and self-similarity. Our experiments show that our approach successfully transfers texture and geometric details to the input 3D asset, outperforming baselines both qualitatively and quantitatively. We also show that traditional metrics are not suitable for evaluating the task due to their inability of focusing on local details and comparing dissimilar inputs, in absence of ground truth data. We thus evaluate appearance transfer quality with a GPT-based system objectively ranking outputs, ensuring robust and human-like assessment, as further confirmed by our user study. Beyond showcased scenarios, our method is general and could be extended to different types of diffusion models and guidance functions.

# :newspaper: News

- [2025-09] GuideFlow3D is accepted to **NeurIPS 2025** 🔥 See you in San Diego!

## 🚧 Code Release

⏳ Code and data will be released by the end of November! Stay tuned for updates. 

## 📧 Contact
If you have any questions regarding this project, please use the github issue tracker or contact Sayan Deb Sarkar (sdsarkar@stanford.edu).

# :page_facing_up: Citation

```bibtex
@inproceedings{sayandsarkar_2025_guideflow3d,
      author = {Deb Sarkar, Sayan and Stekovic, Sinisa and Lepetit, Vincent and Armeni, Iro},
      title = {GuideFlow3D: Optimization-Guided Rectified Flow For 3D Appearance Transfer},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2025},
}
```