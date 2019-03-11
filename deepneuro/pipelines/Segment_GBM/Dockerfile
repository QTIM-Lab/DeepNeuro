FROM qtimlab/deepneuro:latest
LABEL maintainer "Andrew Beers <andrew_beers@alumni.brown.edu>"

# Copy in models -- do this with Python module in future.
RUN mkdir -p /home/DeepNeuro/deepneuro/load/Segment_GBM
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/Segment_GBM_Wholetumor_Model.h5 "https://www.dropbox.com/s/bnbdi1yogq2yye3/GBM_Wholetumor_Public.h5?dl=1"
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_GBM/Segment_GBM_Enhancing_Model.h5 "https://www.dropbox.com/s/hgsqi0vj7cfuk1g/GBM_Enhancing_Public.h5?dl=1"

RUN mkdir -p /home/DeepNeuro/deepneuro/load/SkullStripping
RUN wget -O /home/DeepNeuro/deepneuro/load/SkullStripping/Skullstrip_MRI_Model.h5 "https://www.dropbox.com/s/cucffmytzhp5byn/Skullstrip_MRI_Model.h5?dl=1"

# Commands at startup.
WORKDIR "/"
RUN chmod 777 /home/DeepNeuro/entrypoint.sh
ENTRYPOINT ["/home/DeepNeuro/entrypoint.sh"]
