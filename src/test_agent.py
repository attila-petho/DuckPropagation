import os
import gym
import time
from gym_duckietown.simulator import Simulator
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from wrappers import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


algo            = "A2C"                 # name of RL algo
map_name        = "straight_road"       # map used for training
steps           = "5e5"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
maxsteps        = 200                   # number of steps to take in the test environment

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"
model_name  = f"{algo}_{steps}steps_lr{LR}_{color}_FS{FS}_{action_wrapper}"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_dir    = f"../models/{map_name}/{algo}/"

# Create env
env = Simulator(
        seed=123,                       # random seed
        map_name=map_name,
        max_steps=50001,                # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=3,       # start close to straight
        full_transparency=True,
        distortion=True,
    )

# Wrap environment
env = Monitor(env)
env = ResizeFrame(env, 84)
env = CropFrame(env, 24)
if color_segment:                       # GrayScale and ColorSegment wrappers should not be used at the same time!
        env = ColorSegmentFrame(env)
else:
        env = GrayScaleFrame(env)
env = StackFrame(env, FS)
if action_wrapper == "heading":         # Action wrappers ("heading" can be given a 'type' parameter)
        env = Heading2WheelVelsWrapper(env)
elif action_wrapper == "leftrightbraking":
        env = LeftRightBraking2WheelVelsWrapper(env)
else:
        print("Invalid action wrapper. Using default actions.")
env = DtRewardPosAngle(env)
env = DtRewardVelocity(env)
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

# Load trained model
model = A2C.load(save_dir + model_name, print_system_info=True)

# Test the agent
obs = env.reset()

for step in range(maxsteps):
    action, _ = model.predict(obs, deterministic=True)
    print("\nACTION: ", action)
    observation, reward, done, misc = env.step(action)
    print("Reward: ", reward)
    env.render()
    time.sleep(0.034)
    if done:
        env.render()
        env.reset()
        print("Done.")

env.close()
