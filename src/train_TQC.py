import os
import gym
from gym_duckietown.simulator import Simulator
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import *
from timeit import default_timer as timer


# Arguments
map_name        = "zigzag_dists"        # map used for training
steps           = "1e6"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
checkpoint_freq = 100000                # Checkpoint save frequency
seed            = 123                   # Seed for pseudo random generators

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"

# Create directories for logs
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_dir = f"../logs/{map_name}/TQC_log/"
os.makedirs(log_dir, exist_ok=True)
tensorboard_log = f"../tensorboard/{map_name}/"
os.makedirs(tensorboard_log, exist_ok=True)

# Create environment
env = Simulator(
        seed=seed,                      # random seed
        map_name=map_name,
        max_steps=501,                  # we don't want the gym to reset itself
        domain_rand=1,                  # domain randomization ON
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

model = TQC(
        "CnnPolicy",
        env,
        learning_rate = float(LR),
        verbose = 1,
        tensorboard_log = tensorboard_log,
        seed = seed
        )

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
        save_freq = checkpoint_freq,
        save_path = f'../models/{map_name}/TQC/checkpoints/TQC_{steps}steps_lr{LR}_{color}_FS{FS}_DR',
        name_prefix = 'step_')

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        tb_log_name = f"TQC_{steps}steps_lr{LR}_{color}_FS{FS}_DR"
        )

# Save trained model
model.save(f"../models/{map_name}/TQC/TQC_{steps}steps_lr{LR}_{color}_FS{FS}_DR")
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {int((end-start)/60)} mins\n")
