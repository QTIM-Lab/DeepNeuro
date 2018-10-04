FROM nvidia/cuda:9.0-devel-ubuntu16.04
LABEL maintainer "Andrew Beers <andrew_beers@alumni.brown.edu>"

ARG TENSORFLOW_VERSION=1.11.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.2.2

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#   cat /proc/driver/nvidia/version && \
#   echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#   nvcc -V && \
#   echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
# RUN apt-get update && apt-get install -y \
# 		bc \
# 		build-essential \
# 		cmake \
# 		curl \
# 		g++ \
# 		gfortran \
# 		git \
# 		libffi-dev \
# 		libfreetype6-dev \
# 		libhdf5-dev \
# 		libjpeg-dev \
# 		liblcms2-dev \
# 		libopenblas-dev \
# 		liblapack-dev \
# 		libopenjpeg5 \
# 		libpng12-dev \
# 		libssl-dev \
# 		libtiff5-dev \
# 		libwebp-dev \
# 		libzmq3-dev \
# 		nano \
# 		pkg-config \
# 		python-dev \
# 		software-properties-common \
# 		unzip \
# 		vim \
# 		wget \
# 		zlib1g-dev \
# 		qt5-default \
# 		libvtk6-dev \
# 		zlib1g-dev \
# 		libjpeg-dev \
# 		libwebp-dev \
# 		libpng-dev \
# 		libtiff5-dev \
# 		libjasper-dev \
# 		libopenexr-dev \
# 		libgdal-dev \
# 		libdc1394-22-dev \
# 		libavcodec-dev \
# 		libavformat-dev \
# 		libswscale-dev \
# 		libtheora-dev \
# 		libvorbis-dev \
# 		libxvidcore-dev \
# 		libx264-dev \
# 		yasm \
# 		libopencore-amrnb-dev \
# 		libopencore-amrwb-dev \
# 		libv4l-dev \
# 		libxine2-dev \
# 		libtbb-dev \
# 		libeigen3-dev \
# 		python-dev \
# 		python-tk \
# 		python-numpy \
# 		python3-dev \
# 		python3-tk \
# 		python3-numpy \
# 		ant \
# 		default-jdk \
# 		doxygen \
# 		&& \
# 	apt-get clean && \
# 	apt-get autoremove && \
# 	rm -rf /var/lib/apt/lists/* && \
# # Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
# 	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

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
RUN	apt-get clean && \
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

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
# RUN apt-get update && apt-get install -y \
# 		python-numpy \
# 		python-scipy \
# 		python-nose \
# 		python-h5py \
# 		python-skimage \
# 		python-matplotlib \
# 		python-pandas \
# 		python-sklearn \
# 		python-sympy \
# 		&& \
# 	apt-get clean && \
# 	apt-get autoremove && \
# 	rm -rf /var/lib/apt/lists/*

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
# RUN pip --no-cache-dir install \
	# https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
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

# Environmental Variables
ENV PATH "$PATH:/opt/slicer"
ENV PATH "$PATH:/usr/local/ANTs.2.1.0.Debian-Ubuntu_X64"

# Install DeepNeuro. Scikit-image has installation problems with EasyInstall and setup.py
RUN git clone https://github.com/QTIM-Lab/DeepNeuro /home/DeepNeuro
WORKDIR /home/DeepNeuro

# Copy in models -- do this with Python module in future.
RUN mkdir -p /home/DeepNeuro/deepneuro/load/Segment_GBM
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/Segment_GBM_Wholetumor_Model.h5 "https://www.dropbox.com/s/bnbdi1yogq2yye3/GBM_Wholetumor_Public.h5?dl=1"
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/Segment_GBM_Enhancing_Model.h5 "https://www.dropbox.com/s/hgsqi0vj7cfuk1g/GBM_Enhancing_Public.h5?dl=1"

RUN mkdir -p /home/DeepNeuro/deepneuro/load/SkullStripping
RUN wget -O /home/DeepNeuro/deepneuro/load/SkullStripping/Skullstrip_MRI_Model.h5 "https://www.dropbox.com/s/cucffmytzhp5byn/Skullstrip_MRI_Model.h5?dl=1"

# TODO: Check if Docker has solved this checkpointing problem.
RUN echo 53
RUN git pull
RUN git branch -f data_refactoring origin/data_refactoring
RUN git checkout data_refactoring
RUN python3 /home/DeepNeuro/setup.py develop

# Commands at startup.
WORKDIR "/"
RUN chmod 777 /home/DeepNeuro/entrypoint.sh
ENTRYPOINT ["/home/DeepNeuro/entrypoint.sh"]
