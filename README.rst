.. figure:: ./package_resources/logos/DeepNeuro_alt.PNG?raw=true
   :alt: DeepNeuro

   Alt text
|Build Status|

DeepNeuro
=========

A deep learning python package for neuroimaging data. Focused on validated command-line tools you
can use today. Created by the Quantitative Tumor Imaging Lab at the Martinos Center (Harvard-MIT
Program in Health, Sciences, and Technology / Massachusetts General Hospital).

Table of Contents
-----------------

-  `:question: About <#about>`__
-  `:floppy\_disk: Installation <#installation>`__
-  `:mortar\_board: Tutorials <#tutorials>`__
-  `:gift: Modules <#modules>`__
-  `:speech\_balloon: Contact <#contact>`__
-  `:mega: Citation <#citation>`__
-  `:yellow\_heart: Acknowledgements <#acknowledgements>`__

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

This package is under active development, but we encourage users to both try the modules with
pre-trained modules highlighted below, and try their hand at making their own DeepNeuro modules
using the tutorials below.

Installation
------------

1. Install Docker from Docker's website here: https://www.docker.com/get-started. Follow
   instructions on that link to get Docker set up properly on your workstation.

2. Install the Docker Engine Utility for NVIDIA GPUs, AKA nvidia-docker. You can find installation
   instructions at their Github page, here: https://github.com/NVIDIA/nvidia-docker

3. Pull the DeepNeuro Docker container from
   https://hub.docker.com/r/qtimlab/deepneuro\_segment\_gbm/. Use the command "docker pull
   qtimlab/deepneuro"

4. If you want to run DeepNeuro outside of a Docker container, you can install the DeepNeuro Python
   package locally using the pip package manager. On the command line, run ``pip install deepneuro``

Tutorials
---------

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

   <p align="center">

.. raw:: html

   </p>

   <p align="center">

.. raw:: html

   </p>

Modules
-------

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

   <p align="center">

.. raw:: html

   </p>

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
to graphics processing units. The DeepNeuro project is also indebted to the following Github
repository for the 3D UNet by user ellisdg, which formed the original kernel for much of its code in
early stages. Long live open source deep learning :)

.. |Build Status| image:: https://travis-ci.org/QTIM-Lab/DeepNeuro.svg?branch=master
   :target: https://travis-ci.org/QTIM-Lab/DeepNeuro
