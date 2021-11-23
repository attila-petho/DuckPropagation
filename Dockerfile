FROM python:3.8-buster

RUN mkdir -p /home/app

COPY src /home/app
COPY gym-duckietown /home/app/duckietown



##RUN python3 -m pip install -U "pip>=21"
WORKDIR "/home/app/duckietown"

## first install the ones that do not change
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install freeglut3-dev -y
RUN apt-get install xvfb -y

RUN python3 -m pip install -e .
## RUN python3 -m pip install -r requirements.txt
## RUN python3 -m pip install pyglet==1.5.15
##RUN pip install stable-baselines3

RUN python3 -m pip install stable-baselines3 
RUN apt-get install python-pyglet -y
## RUN python3 -m pip install gym
## RUN python3 -m pip install seaborn


WORKDIR "/home/app/"

CMD ["python3", "test1.py"]