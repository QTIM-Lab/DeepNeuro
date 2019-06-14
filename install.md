# Installation

You can install and use DeepNeuro in three different ways:

- You can install DeepNeuro as a Python package in your local installation of Python using the _pip_ package installer. This will install DeepNeuro and all of its Python dependencies, but will not install links to external imaging packages that DeepNeuro can take advantage of for data processing.
- You can install DeepNeuro via a Docker container. This container will include all external packages necessary to do the full range of data processing in DeepNeuro.
- You can isntall DeepNeuro via a Singularity container, which is very similar to installing via a Docker container.


## Python Installation

You can install DeepNeuro using the _pip_ package manager using the following line:

`pip install deepneuro`

To verify your installation, start Python and run the following line:

`from deepneuro.core import *`

If you install DeepNeuro via _pip_, you will also have access to command-line utilities that allow you to run DeepNeuro's modules. You can see more details about these utilities in the [documentation](documentation.md) page.

## Docker Installation

You can also run DeepNeuro via a Docker container. In practice, Docker containers are tools that allow you to install all of the requirements of a package via one program called Docker. The installation of a package you receive via Docker is a carbon copy of the package that its developers work with, and so it is gaurunteed to work as they expect it to. Docker containers can install more than just Python packages, so installing via this method will also install

To install DeepNeuro via Docker, you first need to install Docker. Read instructions on how to install Docker via [their official website](https://docs.docker.com/v17.12/install/). You will most likely want to install the "Community Edition", although the "Enterprise Edition" will also work.

You will also have to install _nvidia-docker_ to use DeepNeuro. nvidia-docker is an extension to Docker that allows you use NVIDIA GPUs with Docker. DeepNeuro currently requires NVIDIA GPUs to run. You can find instructions on how to install _nvidia-docker_ on [their Github Repository](https://github.com/NVIDIA/nvidia-docker).

Once you have installed Docker, you can install DeepNeuro via the command line.

`docker pull qtimlab/deepneuro`

To verify your installation, run the following line in the command prompt:

`docker run --runtime=nvidia qtimlab/deepneuro python3 -c 'from deepneuro.core import *'`

### Runnning modules with the DeepNeuro Docker

For details on how to run modules with Docker, please see the [Modules](modules.md) page.

### Writing code with the DeepNeuro Docker

For details on how to run modules with Docker, please see the [Modules](modules.md) page.