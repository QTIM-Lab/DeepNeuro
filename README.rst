.. figure:: ./package_resources/logos/DeepNeuro_alt.PNG?raw=true
   :alt: DeepNeuro

   Alt text
|Build Status|

DeepNeuro
=========

A deep learning python package for neuroimaging data. Focused on validated command-line tools you
can use today. Created by the Quantitative Tumor Imaging Lab at the Martinos Center (Harvard-MIT
Program in Health, Sciences, and Technology / Massachussets General Hospital).

Table of Contents
-----------------

-  `About <#about>`__
-  `Installation <#installation>`__
-  `Modules <#modules>`__
-  `Contact <#contact>`__
-  `Citation <#citation>`__
-  `Acknowledgements <#acknowledgements>`__

About
-----

DeepNeuro is an open-source toolset of deep learning applications for neuroimaging. We have several
goals for this package:

-  Provide easy-to-use command line tools for neuroimaging using deep learning.
-  Create Docker containers for each tool and all out-of-package pre-processing steps, so they can
   each can be run without having install prerequisite libraries.
-  Provide freely available deep learning models trained on a wealth of neuroimaging data.
-  Provide training scripts and links to publically-available data to replicate the results of
   DeepNeuro's models.
-  Provide implementations of popular models for medical imaging data, and pre-processed datasets
   for educational purposes.

This package is in an initial testing phase, and will be released soon. Currently, out glioblastoma
segmentation package is available -- see details below for installation and usage instructions.

Installation
------------

1. Install the Docker Engine Utility for NVIDIA GPUs, AKA nvidia-docker. You can find installation
   instructions at their Github page, here: https://github.com/NVIDIA/nvidia-docker

2. Pull the DeepNeuro Docker container from
   https://hub.docker.com/r/qtimlab/deepneuro\_segment\_gbm/. Use the command "docker pull
   qtimlab/deepneuro-segment\_gbm"

3. If you want to inspect the code, or run your Docker container with an DeepNeuro's python wrappers
   and command line tools, clone this repository ("git clone
   https://github.com/QTIM-Lab/DeepNeuro"), and install with the command "python setup.py install"
   in the directory you just cloned in to.

Modules
-------

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

   <p align="center">

.. raw:: html

   </p>

Contact
-------

DeepNeuro is under active development, and you may run into errors or want additional features. Send
any questions or requests for methods to abeers@mgh.harvard.edu. You can also submit a Github issue
if you run into a bug.

Citation
--------

If you use DeepNeuro in your published work, please cite:

Beers, A., Brown, J., Chang, K., Hoebel, K., Gerstner, E., Rosen, B., & Kalpathy-Cramer, J. (2018).
DeepNeuro: an open-source deep learning toolbox for neuroimaging. arXiv preprint arXiv:1808.04589.

@article{beers2018deepneuro, title={DeepNeuro: an open-source deep learning toolbox for
neuroimaging}, author={Beers, Andrew and Brown, James and Chang, Ken and Hoebel, Katharina and
Gerstner, Elizabeth and Rosen, Bruce and Kalpathy-Cramer, Jayashree}, journal={arXiv preprint
arXiv:1808.04589}, year={2018} }

Acknowledgements
----------------

The Center for Clinical Data Science at Massachusetts General Hospital and the Brigham and Woman's
Hospital provided technical and hardware support for the development of DeepNeuro, including access
to high-powered graphical processing units.

.. |Build Status| image:: https://travis-ci.org/QTIM-Lab/DeepNeuro.svg?branch=master
   :target: https://travis-ci.org/QTIM-Lab/DeepNeuro
