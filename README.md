![Alt text](./package_resources/logos/DeepNeuro_alt.PNG?raw=true "DeepNeuro")

[![Build Status](https://travis-ci.org/QTIM-Lab/DeepNeuro.svg?branch=master)](https://travis-ci.org/QTIM-Lab/DeepNeuro)

# DeepNeuro

A deep learning python package for neuroimaging data. Focused on validated command-line tools you can use today. Created by the Quantitative Tumor Imaging Lab at the Martinos Center (Harvard-MIT Program in Health, Sciences, and Technology / Massachussets General Hospital).

## Table of Contents
- [:question: About](#about)
- [:floppy_disk: Installation](#installation) 
- [:mortar_board: Tutorials](#tutorials)
- [:gift: Modules](#modules)
- [:speech_balloon: Contact](#contact)
- [:mega: Citation](#citation)
- [:yellow_heart: Acknowledgements](#acknowledgements)

## About
DeepNeuro is an open-source toolset of deep learning applications for neuroimaging. We have several goals for this package:

* Provide easy-to-use command line tools for neuroimaging using deep learning.
* Create Docker containers for each tool and all out-of-package pre-processing steps, so they can each can be run without having install prerequisite libraries.
* Provide freely available deep learning models trained on a wealth of neuroimaging data.
* Provide training scripts and links to publically-available data to replicate the results of DeepNeuro's models.
* Provide implementations of popular models for medical imaging data, and pre-processed datasets for educational purposes.

This package is under active development, but we encourage users to both try the modules with pre-trained modules highlighted below, and try their hand at making their own DeepNeuro modules using the tutorials below.

## Installation

1. Install Docker from Docker's website here: https://www.docker.com/get-started. Follow instructions on that link to get Docker set up properly on your workstation.

2. Install the Docker Engine Utility for NVIDIA GPUs, AKA nvidia-docker. You can find installation instructions at their Github page, here: https://github.com/NVIDIA/nvidia-docker

3. Pull the DeepNeuro Docker container from https://hub.docker.com/r/qtimlab/deepneuro_segment_gbm/. Use the command "docker pull qtimlab/deepneuro"

4. If you want to run DeepNeuro outside of a Docker container, you can install the DeepNeuro Python package locally using the pip package manager. On the command line, run ```pip install deepneuro```

## Tutorials

<p align="center">
<a href="https://colab.research.google.com/github/QTIM-Lab/DeepNeuro/blob/master/notebooks/Preprocess_and_Augment.ipynb">
<img src="./notebooks/resources/train_preprocess_icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

<p align="center">
<a href="https://colab.research.google.com/github/QTIM-Lab/DeepNeuro/blob/master/notebooks/Train_Model.ipynb">
<img src="./notebooks/resources/train_model_icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

<p align="center">
<a href="https://colab.research.google.com/github/QTIM-Lab/DeepNeuro/blob/master/notebooks/Run_Inference.ipynb">
<img src="./notebooks/resources/model_inference_icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

## Modules

<p align="center">
<a href="https://github.com/QTIM-Lab/DeepNeuro/tree/master/deepneuro/pipelines/Segment_GBM">
<img src="./deepneuro/pipelines/Segment_GBM/resources/icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

<p align="center">
<a href="https://github.com/QTIM-Lab/DeepNeuro/tree/master/deepneuro/pipelines/Skull_Stripping">
<img src="./deepneuro/pipelines/Skull_Stripping/resources/icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

<p align="center">
<a href="https://github.com/QTIM-Lab/DeepNeuro/tree/master/deepneuro/pipelines/Segment_Brain_Mets">
<img src="./deepneuro/pipelines/Segment_Brain_Mets/resources/icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

<p align="center">
<a href="https://github.com/QTIM-Lab/DeepNeuro/tree/master/deepneuro/pipelines/Ischemic_Stroke">
<img src="./deepneuro/pipelines/Ischemic_Stroke/resources/icon.png?raw=true" width="684" alt=""></img>
</a>
</p>

## Contact

DeepNeuro is under active development, and you may run into errors or want additional features. Send any questions or requests for methods to abeers@mgh.harvard.edu. You can also submit a Github issue if you run into a bug.

## Citation

If you use DeepNeuro in your published work, please cite:

Beers, A., Brown, J., Chang, K., Hoebel, K., Gerstner, E., Rosen, B., & Kalpathy-Cramer, J. (2018). <a href="https://arxiv.org/pdf/1808.04589.pdf">DeepNeuro: an open-source deep learning toolbox for neuroimaging</a>. arXiv preprint arXiv:1808.04589.

@article{beers2018deepneuro,
  title={DeepNeuro: an open-source deep learning toolbox for neuroimaging},
  author={Beers, Andrew and Brown, James and Chang, Ken and Hoebel, Katharina and Gerstner, Elizabeth and Rosen, Bruce and Kalpathy-Cramer, Jayashree},
  journal={arXiv preprint arXiv:1808.04589},
  year={2018}
}

## Acknowledgements

The Center for Clinical Data Science at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware support for the development of DeepNeuro, including access to high-powered graphical processing units. The DeepNeuro project is also indebted to the following <a href="https://github.com/ellisdg/3DUnetCNN">Github repository</a> for the 3D UNet by user ellisdg, which formed the original kernel for much of its code in early stages. Long live open source deep learning :)
