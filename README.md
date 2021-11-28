# DuckPropagation
In this project we are training Deep Reinforcement Learning agents to drive small robots called Duckiebots in the Duckietown environment. There are four challenges in the environment:

- LF - simple lane following
- LFV - lane following with vehicles
- LFI - lane following with intersections
- LFVI - lane following with vehicles and intersections

In order to conquer these challenges, autonomous driving agents are first trained in a simulator (gym-duckietown) and then the trained agents performance are also tested in the real environment on real Duckiebots.


## Authors
Attila Pethő

Gyula Faragó

Olivér Farkas


## Installation
Requirements:
- Python 3.6+
- OpenAI Gym
- gym-duckiwtown
- Numpy
- Pyglet
- PyYAML
- cloudpicke
- Pytorch
- Stable Baselines 3
- opencv
- pillow
- Tensorboard

## Usage
TODO

## Preprocessing
The simulator produces 640x480 RBG images that look like this:

<img src="src/images/raw_obs/20211023_191012_1.jpg" width="640px">

The first step is to preprocess these images for the CNN. This step is necessary because feeding the original images makes the training process much slower and it would also be a waste of resources because the network can learn from much smaller images just as well. So for the optimal learning process we do the following steps:

**1. Resizing**

**2. Cropping**

**3. Color segmentation or Grayscaling**

**4. Normalization**

**5. Frame stacking** 

These preprocessing steps require the use of so-called wrappers, which basically wrap around the enviroment, take the observations and convert them according to their purpose. We created the following wrappers for these tasks:

#### ResizeFrame
With this wrapper we are downscaling the images from their original size (480x640x3) to (84x84x3). The smaller dimension makes the training of the neural network faster, and it still carries enough information for efficient training.

#### CropFrame
In this wrapper we are cropping the useless information from the image, in our case it's the part above the horizon.

#### GrayScaleFrame
Training time can be reduced by using grayscale images instead of RGB, while keeping the important information of the images. This wrapper should not be used in conjunction with the ColorSegmentFrame wrapper.

#### ColorSegmentFrame
Here we are segmenting the different parts of the image so we can feed the neural network more useful information. The segmentation is done using intervals, we assigned the red channel for the white line, and the green channel for the yellow line. For lane following only these two information are useful for the neural network, so we assign black for everything else.

#### NormalizeFrame
To make the training of the CNN easier and faster this wrapper normalizes the pixel values in the interval of [0,1]. Altough we implemented this wrapper, it is not used, because stable baselines does the input normalization automatically.

#### StackFrame
For better quality in training and more information we are concatenating the last **n frames** to form a time series, so the agent can percieve dynamic changes in its environment.

## Actions
TODO

## Rewards
TODO

## Training
TODO: algos
For the training we used the Stable Baselines 3 library, which contains several implementations of state-of-the-art RL algorithms. The wrappers are tested with an A2C agent with default settings and 0.00005 learning rate for 1 million steps on the 'straight_road' map. It took 139 minutes on a GTX1060 OC 6GB. The trained model and the tensorboard log of the first training can be found in the corresponding folders.

## Evaluation
TODO

## File Structure

- `src/`             - The source folder.
- `src/images/`      - Output samples of the wrappers are stored here.
- `src/tensorboard/` - Tensorboard logs folder.
- `src/test1.py`     - Test script used for checking the wrappers.
- `src/train_A2C.py` - Training script for the A2C agent.
- `src/train_PPO.py` - Training script for the PPO agent.
- `src/wrappers.py`  - Contains the wrappers.
- `models/`          - Save location for the trained models.
- `logs/`            - SB3 logs folder.
