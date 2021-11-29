FROM python:3.8-buster

RUN mkdir -p /home/app

COPY src /home/app


RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git

RUN git clone https://github.com/duckietown/gym-duckietown.git --branch daffy
RUN pip install -r gym-duckietown/requirements.txt
RUN pip install -e gym-duckietown

#RUN python3 -m pip install -e .


RUN python3 -m pip install stable-baselines3 



WORKDIR "/home/app/"

CMD ["python3", "test1.py"]