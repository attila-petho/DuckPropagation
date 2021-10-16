import gym
import numpy as np

class CropFrame(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        super(CropFrame, self).__init__(env)
        self.crop = crop
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=(59, 84, 1),
                                                dtype=env.observation_space.dtype)

def observation(self, obs):
    return CropFrame.process(obs)

@staticmethod
def process(frame):
    img = frame
    x_t = img[25:84, :]
    return x_t.astype(np.uint8)