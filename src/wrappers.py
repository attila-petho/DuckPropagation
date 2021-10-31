import gym
import numpy as np
from gym import spaces

# Constants
sensitivity_yellow = 100
sensitivity_white = 50
colorcode_yellow = [255,255,0]
colorcode_white = [255,255,255]
threshold_yellow = [colorcode_yellow[0] - sensitivity_yellow, colorcode_yellow[1] - sensitivity_yellow, colorcode_yellow[2] + sensitivity_yellow]
threshold_white = [colorcode_white[0] - sensitivity_white, colorcode_white[1] - sensitivity_white, colorcode_white[2] - sensitivity_white]


class ResizeFrame(gym.ObservationWrapper):
    '''
    Resizes the observation to (size x size x channels)
    '''
    def __init__(self, env=None, size=84):
        super(ResizeFrame, self).__init__(env)
        self.shape = (size, size, self.observation_space.shape[2])
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class CropFrame(gym.ObservationWrapper):
    '''
    Crops the top n pixels of the observation
    '''
    def __init__(self, env, crop=24):
        super(CropFrame, self).__init__(env)
        self.crop = crop
        self.obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, 
                                            high=255,
                                            shape=(self.obs_shape[0]-crop, self.obs_shape[1], self.obs_shape[2]),
                                            dtype=env.observation_space.dtype)

    def observation(self, obs):
        img = obs[self.crop:self.obs_shape[0],:,:]
        return img.astype(np.uint8)


class GrayScaleFrame(gym.ObservationWrapper):
    '''
    Converts RGB images to Grayscale
    '''
    def __init__(self, env):
        super(GrayScaleFrame, self).__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(obs_shape[0], obs_shape[1], 1),
                                            dtype=np.uint8)

    def observation(self, observation):
        import cv2
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, -1)
        return observation


class ColorSegmentFrame(gym.ObservationWrapper):
    '''
    Separates the yellow and white parts of the image to channels Red and Green
    '''
    def __init__(self, env):
        super(ColorSegmentFrame, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(obs_shape[0], obs_shape[1], 3),
                                            dtype=np.uint8)

    def observation(self, observation):
        for x in observation:
            for y in x:
                if((y[0] - threshold_yellow[0] > 0) and (y[1] - threshold_yellow[1] > 0) and (y[2] - threshold_yellow[2] < 0)):  #Is yellow?
                    y[0] = 255
                    y[1] = 0
                    y[2] = 0
                elif((y[0] - threshold_white[0] > 0) and (y[1] - threshold_white[1] > 0) and (y[2] - threshold_white[2] > 0)): #Is white?
                    y[0] = 0
                    y[1] = 255
                    y[2] = 0
                else:
                    y[0] = 0
                    y[1] = 0
                    y[2] = 0
        return observation


class NormalizeFrame(gym.ObservationWrapper):
    '''
    Converts obervations to floats between 0 and 1
    '''
    def __init__(self, env=None):
        super(NormalizeFrame, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=obs_shape,
                                            dtype=np.float32)

    def observation(self, obs):
        return obs / 255


class StackFrame(gym.ObservationWrapper):
    '''
    Stacks the last n observations
    '''
    def __init__(self, env, nframes=4):
        super(StackFrame, self).__init__(env)
        self.nframes = nframes
        from collections import deque
        self.frames = deque([], maxlen=nframes)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=((nframes,) + shp),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.nframes):
            self.frames.append(obs)
        return np.array(self.frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, done, info

