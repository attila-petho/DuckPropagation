import gym
import numpy as np
from gym import spaces


class ResizeFrame(gym.ObservationWrapper):
    '''
    Resizes the image from (480x640x3) to (84x84x3)
    '''
    def __init__(self, env=None, shape=(84, 84, 3)):
        super(ResizeFrame, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class CropFrame(gym.ObservationWrapper):
    '''
    Crops the top 24 pixels of the (84x84x3) image
    '''
    def __init__(self, env, crop=True):
        super(CropFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, 
                                            high=255,
                                            shape=(60, 84, 3),
                                            dtype=env.observation_space.dtype)

    def observation(self, obs):
        img = obs[24:84,:,:]                # TODO: crop in 3d!!
        return img.astype(np.uint8)


class GrayScaleFrame(gym.ObservationWrapper):
    '''
    Description TODO
    '''
    ...


class ColorSegmentFrame(gym.ObservationWrapper):
    '''
    Description TODO
    '''
    ...


class NormalizeFrame(gym.ObservationWrapper):
    '''
    Description TODO
    '''
    ...


class StackFrame(gym.ObservationWrapper):
    '''
    Description TODO
    '''
    ...

