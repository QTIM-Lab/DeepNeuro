FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
LABEL maintainer "Andrew Beers <andrew_beers@alumni.brown.edu>"

ARG TENSORFLOW_VERSION=1.2.1
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.0.6

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \
#	echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libopenjpeg2 \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		python-dev \
		python-tk \
		python-numpy \
		python3-dev \
		python3-tk \
		python3-numpy \
		ant \
		default-jdk \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
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
	python -m ipykernel.kernelspec

# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Install Additional Packages for DeepNeuro
RUN apt-get update -y
RUN apt-get install graphviz -y
RUN pip install pydot==1.1.0
RUN pip install pandas --upgrade 
RUN pip install numexpr --upgrade
RUN pip install nibabel pydicom

# Install Slicer
 RUN SLICER_URL="http://download.slicer.org/bitstream/561384" && \
    curl -v -s -L $SLICER_URL | tar xz -C /tmp && \
    mv /tmp/Slicer* /opt/slicer

# Install ANTS
WORKDIR /home
RUN wget "https://github.com/stnava/ANTs/releases/download/v2.1.0/Linux_Ubuntu14.04.tar.bz2" && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --force-yes bzip2 && \
  tar -C /usr/local -xjf Linux_Ubuntu14.04.tar.bz2 && \
  rm Linux_Ubuntu14.04.tar.bz2

# Install NeuroDebian
RUN wget -O- http://neuro.debian.net/lists/trusty.us-nh.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9
RUN apt-get update

# Install FSL with NeuroDebian
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --force-yes fsl-5.0-complete

# Environmental Variables
ENV PATH "$PATH:/opt/slicer"
ENV PATH "$PATH:/usr/local/ANTs.2.1.0.Debian-Ubuntu_X64"

# Setup Scripts
RUN echo "source /usr/share/fsl/5.0/etc/fslconf/fsl.sh" >> ~/.bashrc

RUN git clone https://github.com/QTIM-Lab/DeepNeuro /home/DeepNeuro
WORKDIR /home/DeepNeuro
RUN python /home/DeepNeuro/setup.py develop

# Copy in models
RUN mkdir -p /home/DeepNeuro/deepneuro/load/Segment_GBM
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/wholetumor.h5 "https://www.dropbox.com/s/74tjx14ue11rc0q/wholetumor.h5?dl=1"
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/enhancing.h5 "https://www.dropbox.com/s/usdal6cbkw3bceu/enhancingtumor_BRATS_submission.h5?dl=1"

# TODO: Check if Docker has solved this checkpointing problem.
RUN echo 22
RUN git pull

# Commands at startup.
WORKDIR "/home/DeepNeuro/misc/DeepInfer/Segment_GBM/"
RUN chmod 777 /home/DeepNeuro/misc/DeepInfer/Segment_GBM/entrypoint.sh
ENTRYPOINT ["/home/DeepNeuro/misc/DeepInfer/Segment_GBM/entrypoint.sh"]
