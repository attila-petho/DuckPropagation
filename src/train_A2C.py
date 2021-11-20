import os
import gym
from gym_duckietown.simulator import Simulator
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import *
from timeit import default_timer as timer


# Arguments
map_name        = "straight_road"       # map used for training
steps           = "5e5"                 # train for 500k steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 4                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
checkpoint_freq = 100000                # Checkpoint save frequency

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"

# Create directories for logs
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_dir = f"../logs/{map_name}/A2C_log/"
os.makedirs(log_dir, exist_ok=True)
tensorboard_log = f"../tensorboard/{map_name}/"
os.makedirs(tensorboard_log, exist_ok=True)

# Create environment
env = Simulator(
        seed=123,                       # random seed
        map_name=map_name,
        max_steps=501,                  # we don't want the gym to reset itself
        domain_rand=0,
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

env.reset()

# Create model
start = timer()

model = A2C(
        "CnnPolicy",
        env,
        learning_rate = float(LR),
        verbose = 1,
        tensorboard_log = tensorboard_log
        )

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
        save_freq = checkpoint_freq,
        save_path = f'../models/{map_name}/A2C/checkpoints/A2C_{steps}steps_lr{LR}_{color}_FS{FS}_{action_wrapper}/',
        name_prefix = 'step_')

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        log_interval = 500,
        tb_log_name = f"A2C_{steps}steps_lr{LR}_{color}_FS{FS}_{action_wrapper}"
        )

# Save trained model
model.save(f"../models/{map_name}/A2C/A2C_{steps}steps_lr{LR}_{color}_FS{FS}_{action_wrapper}")
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {(end-start)/60} mins\n")
