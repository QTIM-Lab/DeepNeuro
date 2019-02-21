FROM qtimlab/deepneuro:latest
LABEL maintainer "Andrew Beers <andrew_beers@alumni.brown.edu>"

# Install DeepNeuro. Scikit-image has installation problems with EasyInstall and setup.py
RUN git clone https://github.com/QTIM-Lab/DeepNeuro /home/DeepNeuro
WORKDIR /home/DeepNeuro

# Copy in models -- do this with Python module in future.
RUN mkdir -p /home/DeepNeuro/deepneuro/load/Segment_Ischemic_Stroke
RUN wget -O /home/DeepNeuro/deepneuro/load/Segment_Ischemic_Stroke/Ischemic_Stroke_Model.h5 "https://www.dropbox.com/s/4qpxvfac204xzhf/Ischemic_Stroke_Segmentation_Model.h5?dl=1"

# TODO: Check if Docker has solved this checkpointing problem.
RUN echo 60
RUN git pull
RUN python3 /home/DeepNeuro/setup.py develop

# Commands at startup.
WORKDIR "/"
RUN chmod 777 /home/DeepNeuro/entrypoint.sh
ENTRYPOINT ["/home/DeepNeuro/entrypoint.sh"]
