FROM ufoym/deepo:pytorch-py36-cu90

RUN rm -rf /$HOME/DuckPropagation

WORKDIR /$HOME/DuckPropagation

COPY src/* $HOME/DuckPropagation/src/
COPY models/* $HOME/DuckPropagation/models/
COPY tensorboard/* $HOME/DuckPropagation/src/
COPY logs/* $HOME/DuckPropagation/logs/
COPY env_setup.sh $HOME/DuckPropagation/
COPY env.yml/* $HOME/DuckPropagation/


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

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["/bin/bash"] 