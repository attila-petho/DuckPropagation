FROM ufoym/deepo:pytorch-py36-cu90
# TODO: try a duckietown base

WORKDIR /DuckPropagation

# Install deps and create env

RUN python -m pip install --upgrade pip

RUN pip install librosa && \
	pip install unidecode &&\
	pip install inflect && \
	pip install tensorboardX &&\
	pip install tensorflow-gpu &&\ 
	pip install matplotlib==2.1.0 &&\
	pip install torch==1.0.0 &&\
	pip install inflect &&\
	pip install scipy &&\
	pip install pillow &&\
	apt-get update && apt-get  install -y \
	nano \
	tmux \
    htop \
	mc  && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends screen

COPY src/* ./src
COPY models/* ./models
COPY tensorboard/* ./tensorboard
COPY logs/* ./logs
COPY env_setup.sh ./
COPY env.yml/* ./

# Set up port for Tensorboard

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["/bin/bash"] 