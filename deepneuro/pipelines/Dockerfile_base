FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "Andrew Beers <andrew_beers@alumni.brown.edu>"

ARG TENSORFLOW_VERSION=1.11.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.2.2

# Install some dependencies
# Install basic packages and miscellaneous dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    liblapack-dev \
    libopenblas-dev \
    libzmq3-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk

# Install Pillow (PIL) dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libfreetype6-dev \
    libjpeg-dev \
    liblcms2-dev \
    libopenjpeg-dev \
    libpng12-dev \
    libtiff5-dev \
    libwebp-dev \
    zlib1g-dev 

# Install support functions
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    wget \
    cmake

# Cleanup Installs
RUN apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
    update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN pip3 install --upgrade \
    setuptools \
    pip

# Add SNI support to Python
RUN pip3 --no-cache-dir install \
        pyopenssl \
        ndg-httpsclient \
        pyasn1

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
    pip3 --no-cache-dir install \
        Cython \
        ipykernel \
        jupyter \
        path.py \
        Pillow \
        pygments \
        six \
        sphinx \
        wheel \
        zmq \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow
# For specific installations -- TODO, peg a version of Tensorflow.
# RUN pip --no-cache-dir install \
    # https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
# Generic.
RUN pip3 --no-cache-dir install tensorflow-gpu

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Install Additional Packages for DeepNeuro
RUN apt-get update -y
RUN apt-get install graphviz -y
RUN pip3 --no-cache-dir install pydot
RUN pip3 --no-cache-dir install pandas --upgrade 
RUN pip3 --no-cache-dir install numexpr --upgrade
RUN pip3 --no-cache-dir install nibabel pydicom lycon tqdm pynrrd tables imageio matplotlib

# Install Slicer
 RUN SLICER_URL="http://download.slicer.org/bitstream/561384" && \
    curl -v -s -L $SLICER_URL | tar xz -C /tmp && \
    mv /tmp/Slicer* /opt/slicer

# Install ANTS
WORKDIR /home
RUN wget "https://github.com/stnava/ANTs/releases/download/v2.1.0/Linux_Ubuntu14.04.tar.bz2" && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --force-yes bzip2 && \
  tar -C /usr/local -xjf Linux_Ubuntu14.04.tar.bz2 && \
  rm Linux_Ubuntu14.04.tar.bz2

# Python 2.7
WORKDIR /usr/src
ENV PYTHON_VERSION 2.7.10
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
  tar xvzf Python-${PYTHON_VERSION}.tgz && \
  cd Python-${PYTHON_VERSION} && \
  ./configure && \
  make -j$(grep -c processor /proc/cpuinfo) && \
  make install && \
  cd .. && rm -rf Python-${PYTHON_VERSION}*

# Build and install dcmqi

WORKDIR /usr/src
ENV PYTHON_VERSION 2.7.10
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
  tar xvzf Python-${PYTHON_VERSION}.tgz && \
  cd Python-${PYTHON_VERSION} && \
  ./configure && \
  make -j$(grep -c processor /proc/cpuinfo) && \
  make install && \
  cd .. && rm -rf Python-${PYTHON_VERSION}*

WORKDIR /usr/src
RUN git clone https://github.com/QIICR/dcmqi.git && \
  mkdir dcmqi-superbuild && \
  cd dcmqi-superbuild && \
  cmake -DCMAKE_INSTALL_PREFIX=/usr ../dcmqi && \
  make -j$(grep -c processor /proc/cpuinfo)

# Environmental Variables
ENV PATH "$PATH:/opt/slicer"
ENV PATH "$PATH:/usr/local/ANTs.2.1.0.Debian-Ubuntu_X64"

# Install DeepNeuro. Scikit-image has installation problems with EasyInstall and setup.py
RUN git clone https://github.com/QTIM-Lab/DeepNeuro /home/DeepNeuro
WORKDIR /home/DeepNeuro
