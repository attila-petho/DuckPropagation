import gym
import numpy as np
from gym import spaces
from gym_duckietown.simulator import NotInLane
from matplotlib import pyplot as plt
import seaborn
import logging

logger = logging.getLogger(__name__)

# Constants

sensitivity_yellow = 100
sensitivity_white = 70
colorcode_yellow = [255,255,0]
colorcode_white = [255,255,255]
threshold_yellow = [colorcode_yellow[0] - sensitivity_yellow, colorcode_yellow[1] - sensitivity_yellow, colorcode_yellow[2] + sensitivity_yellow]
threshold_white = [colorcode_white[0] - sensitivity_white, colorcode_white[1] - sensitivity_white, colorcode_white[2] - sensitivity_white]

# Observation Wrappers

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
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


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
                                            shape=((shp[0],shp[1],shp[2]*nframes)),
                                            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        self.frames.append(obs)
        stackedobs = np.array(obs, dtype='f')
        for n in range(self.nframes-1):
            self.frames.append(obs)
            stackedobs = np.dstack((stackedobs,obs))
        return stackedobs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        stackedobs = np.array(self.frames[0], dtype='f')
        for n in range(self.nframes-1):
            stackedobs = np.dstack((stackedobs,self.frames[n+1]))
        return stackedobs, reward, done, info


# The following wrappers are imported from the Duckietown-RL repo by kaland313 (further info in the README)

# Action Wrappers

class Heading2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env, heading_type=None):
        super(Heading2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        if self.heading_type == 'heading_trapz':
            straight_plateau_half_width = 0.3333  # equal interval for left, right turning and straight
            self.mul = 1. / (1. - straight_plateau_half_width)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # action = [-0.5 * action + 0.5, 0.5 * action + 0.5]
        if self.heading_type == 'heading_smooth':
            action = np.clip(np.array([1 + action ** 3, 1 - action ** 3]), 0., 1.)  # Full speed single value control
        elif self.heading_type == 'heading_trapz':
            action = np.clip(np.array([1 - action, 1 + action]) * self.mul, 0., 1.)
        elif self.heading_type == 'heading_sine':
            action = np.clip([1 - np.sin(action * np.pi), 1 + np.sin(action * np.pi)], 0., 1.)
        elif self.heading_type == 'heading_limited':
            action = np.clip(np.array([1 + action*0.666666, 1 - action*0.666666]), 0., 1.)
        else:
            action = np.clip(np.array([1 + action, 1 - action]), 0., 1.)  # Full speed single value control
        return action


class LeftRightBraking2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LeftRightBraking2WheelVelsWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return np.clip(np.array([1., 1.]) - np.array(action), 0., 1.)


# Reward Wrappers

class DtRewardPosAngle(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardPosAngle, self).__init__(env)

        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 2.                             # CHANGED: from 1./2. --- THIS SHOULD BE UPSCALED FIRST
        self.scaler_narrow = 0.95                    # scaler for the narrow angle [0,1]
        self.scaler_wide = 1-self.scaler_narrow     # scaler for the wide angle
        self.orientation_reward = 0.

    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle
        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            # In case a previous wrapper already did that, we should not do it again
            if reward >= 0.:
                return -10.

        # print("Dist: {:3.2f} | Angle_deg: {:3.2f}".format(normed_lp_dist, normed_lp_angle))
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        #logger.debug("Angle Narrow: {:4.3f} | Angle Wide: {:4.3f} ".format(angle_narrow_reward, angle_wide_reward))
        self.orientation_reward = self.scale * (self.scaler_narrow*angle_narrow_reward + self.scaler_wide*angle_wide_reward)

        # early_termination_penalty = 0.
        # If the robot leaves the track or collides with an other object it receives a penalty
        # if reward <= -1000.:  # Gym Duckietown gives -1000 for this
        #     early_termination_penalty = -10.
        return reward + self.orientation_reward # + early_termination_penalty

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    @staticmethod
    def gaussian(x, mu=0., sig=1.):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide

class DtRewardVelocity(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardVelocity, self).__init__(env)
        self.velocity_reward = 0.

    def reward(self, reward):
        self.velocity_reward = np.max(self.unwrapped.wheelVels) * 0.2 # CHANGED: from 0.25
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.velocity_reward

    def reset(self, **kwargs):
        self.velocity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['velocity'] = self.velocity_reward
        return observation, self.reward(reward), done, info


class DtRewardWrapperDistanceTravelled(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapperDistanceTravelled, self).__init__(env)
        # gym_duckietown.simulator.Simulator):
        self.prev_pos = None

    def reward(self, reward):
        # Baseline reward is a for each step
        my_reward = 0

        # Get current position and store it for the next step
        pos = self.unwrapped.cur_pos
        prev_pos = self.prev_pos
        self.prev_pos = pos
        if prev_pos is None:
            return 0

        # Get the closest point on the curve at the current and previous position
        angle = self.unwrapped.cur_angle
        curve_point, tangent = self.unwrapped.closest_curve_point(pos, angle)
        prev_curve_point, prev_tangent = self.unwrapped.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            logger.error("self.unwrapped.closest_curve_point(pos, angle) returned None!!!")
            return my_reward
        # Calculate the distance between these points (chord of the curve), curve length would be more accurate
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        try:
            lp = self.unwrapped.get_lane_pos2(pos, self.unwrapped.cur_angle)
            print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f} | DIST: {:3.2f}".format(lp.dist, lp.dot_dir, lp.angle_deg, dist))   # UNCHECK THIS FOR DISTANCE DATA
        except NotInLane:
            my_reward = -10.0 # CHANGED
            return my_reward

        # Dist is negative on the left side of the right lane center and is -0.1 on the lane center.
        # The robot is 0.13 (m) wide, to keep the whole vehicle in the right lane, dist should be > -0.1+0.13/2=-0.035
        # 0.05 is a little less conservative
        if lp.dist < -0.04: # CHANGED: from 0.05
            return my_reward # (could be negative)
        # Check if the agent moved in the correct direction
        if np.dot(tangent, diff) < 0:
            my_reward = -1.0 # CHANGED
            return my_reward

        # Reward is proportional to the distance travelled at each step
        my_reward = 90 * dist # CHANGED: from 50
        if np.isnan(my_reward):
            my_reward = 0.
            logger.error("Reward is nan!!!")
        return my_reward