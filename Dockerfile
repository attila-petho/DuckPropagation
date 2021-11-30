FROM python:3.8-buster

RUN mkdir -p /home/duckietown

COPY src /home/duckietown/src

##RUN python3 -m pip install -U "pip>=21"
WORKDIR "/home/duckietown"

## first install the ones that do not change
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install freeglut3-dev -y
RUN apt-get install xvfb -y
RUN apt-get install git wget -y

RUN pip3 install pyglet
RUN apt-get install python-pyglet -y


RUN git clone https://github.com/duckietown/gym-duckietown.git
WORKDIR "/home/duckietown/gym-duckietown"
RUN pip3 install --upgrade pip
RUN pip3 install -e .



RUN pip3 install seaborn numpy matplotlib
RUN pip3 install stable-baselines3[extra]

## RUN python3 -m pip install -r requirements.txt
## RUN python3 -m pip install pyglet==1.5.15
##RUN pip install stable-baselines3

##RUN python3 -m pip install stable-baselines3 
##RUN apt-get install python-pyglet -y
## RUN python3 -m pip install gym
## RUN python3 -m pip install seaborn


WORKDIR "/home/duckietown/src"

RUN Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
RUN export DISPLAY=:0

CMD ["python3", "/home/duckietown/src/train_A2C.py"]
## CMD ["ls", "/home/duckietown"]