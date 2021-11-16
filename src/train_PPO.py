import os
import gym
from gym_duckietown.simulator import Simulator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wrappers import ResizeFrame, CropFrame, GrayScaleFrame, ColorSegmentFrame, StackFrame
from timeit import default_timer as timer


# Arguments
map_name        = "straight_road"       # map used for training
steps           = "1e6"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 4                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"

# Create directories for logs
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_dir = f"../logs/{map_name}/PPO_log/"
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
        accept_start_angle_deg=3,       # start close to straight
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

env.reset()

# Create model
start = timer()

model = PPO(
        "CnnPolicy",
        env,
        learning_rate=float(LR),
        verbose=1,
        tensorboard_log=tensorboard_log
        )

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=f'../models/{map_name}/PPO/checkpoints/PPO_{steps}steps_lr{LR}_{color}_FS{FS}',
        name_prefix='step_')

# Start training
model.learn(
        total_timesteps=int(float(steps)),
        callback=checkpoint_callback,
        log_interval=500,
        tb_log_name=f"PPO_{steps}steps_lr{LR}_{color}_FS{FS}"
        )

# Save trained model
model.save(f"../models/{map_name}/PPO/PPO_{steps}steps_lr{LR}_{color}_FS{FS}")
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {(end-start)/60} mins\n")
