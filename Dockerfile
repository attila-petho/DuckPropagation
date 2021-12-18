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
	&& apt-get install -y libglfw3-dev libgles2-mesa-dev pkg-config libfontconfig1-dev nano htop mc\
	&& apt-get install -y --no-install-recommends screen\
	&& apt install -y xvfb\
	&& apt-get update

# Set up port for Tensorboard
#EXPOSE 6006

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