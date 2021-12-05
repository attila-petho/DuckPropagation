#FROM ufoym/deepo:pytorch-py36-cu90
#FROM gym-duckietown:daffy-amd64-6.1.29
FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /DuckPropagation

# Install deps and create env

RUN set -xe \
    && apt-get update -y\
    && apt-get install -y wget\
	&& apt-get install -y git\
	&& apt-get install -y python3.8 python3-pip\
	&& apt-get install -y python-opengl\
	&& apt-get install -y libglfw3-dev libgles2-mesa-dev pkg-config libfontconfig1-dev

# TODO: continue: https://github.com/duckietown/gym-duckietown/issues/160
# + REDUNDANT PACKAGES!

RUN apt apt install xvfb
	
RUN pip3 install --upgrade pip

RUN pip3 install librosa && \
	pip3 install unidecode &&\
	pip3 install inflect && \
	pip3 install tensorboardX &&\
	pip3 install tensorflow-gpu &&\ 
	pip3 install matplotlib==3.5.0 &&\
	pip3 install torch==1.10.0 &&\
	pip3 install inflect &&\
	pip3 install scipy &&\
	pip3 install pillow &&\
	apt-get update && apt-get  install -y \
	nano \
	tmux \
    htop \
	mc  && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends screen

# Set up port for Tensorboard
EXPOSE 6006

#RUN useradd -m duckprop
#RUN echo 'duckprop:duckie' | chpasswd

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /DuckPropagation/miniconda3
RUN rm Miniconda3-latest-Linux-x86_64.sh
RUN echo ". /DuckPropagation/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate dtgym" >> ~/.bashrc
SHELL ["/DuckPropagation/miniconda3/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
COPY env.yml .
RUN conda env create -f /DuckPropagation/env.yml
SHELL ["/DuckPropagation/miniconda3/bin/conda", "run", "-n", "dtgym", "/bin/bash", "-c"]
RUN git clone https://github.com/duckietown/gym-duckietown.git
RUN pip3 install -e gym-duckietown

COPY . .

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["/bin/bash"] 