import os
import gym
from gym_duckietown.simulator import Simulator
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from wrappers import ResizeFrame, CropFrame, GrayScaleFrame, ColorSegmentFrame, StackFrame
from timeit import default_timer as timer


start = timer()

# Create directories for logs
map_name = "straight_road"
log_dir = f"../logs/{map_name}/A2C_log/"
os.makedirs(log_dir, exist_ok=True)
tensorboard_log = f"../tensorboard/{map_name}/"
os.makedirs(tensorboard_log, exist_ok=True)

# Create environment
env = Simulator(
        seed=123,                       # random seed
        map_name=map_name,
        max_steps=500001,               # we don't want the gym to reset itself
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
env = GrayScaleFrame(env)               # GrayScale and ColorSegment wrappers should not be used at the same time!
#env = ColorSegmentFrame(env)
env = StackFrame(env, 5)

env.reset()

# Create model
LR = 5e-5                               # Learning Rate: 0.00005

model = A2C(
        "CnnPolicy",
        env,
        learning_rate=int(LR),
        verbose=1,
        tensorboard_log=tensorboard_log
        )

# Start training
steps = 1e6                             # train for 1M steps

model.learn(
        total_timesteps=int(steps),
        log_interval=500,
        tb_log_name=f"A2C_{str(steps)}steps_lr{str(LR)}_GrayS"
        )

# Save trained model
model.save(f"../models/{map_name}/A2C_{str(steps)}steps_lr{str(LR)}_GrayS")
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {(end-start)/60} mins\n")
