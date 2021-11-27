# imports
from gym_duckietown.simulator import Simulator
from stable_baselines3.common.monitor import Monitor
from utils.wrappers import *

def make_env(map_name, log_dir, seed=123, domain_rand=1, color_segment=False, FS=3, action_wrapper="heading"):
    env = Simulator(
            seed=seed,                      # random seed
            map_name=map_name,
            max_steps=501,                  # we don't want the gym to reset itself
            domain_rand=1,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,       # start close to straight
            full_transparency=True,
            distortion=True,
        )

    # Wrap environment
    env = Monitor(env, log_dir)
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
    return env