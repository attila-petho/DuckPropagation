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
    Normalizes the image from the 0-255 interval to 0-1
    '''
    def __init__(self, env, crop=True):
        super(NormalizeFrame, self).__init__(env)
    def observation(self, obs):
        img = obs
        img /= 255
        return img.astype("float32")


class StackFrame(gym.ObservationWrapper):
    '''
    Concatenates images
    img_con = The numpy array in which we store the concatenated images
    i = Number of images we want in a sequence
    '''
    #valahogy ciklusban kéne majd meghívni, pl:
    #img_conc
    #for i = 0 in range(5):
    #    if i == 0:
    #       img_conc = basic observation #simán egy kép
    #    else:
    #       img_conc = StackFrame.observation
    def __init__(self, env, i = 4):
        super(StackFrame, self).__init__(env)
    def observation(img_con, obs):
        if img_con.shape[2] / 3 < i:
            img = obs
            return np.concatenate((img_con, img), axis = 2)
        else:
            return