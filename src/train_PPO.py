import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env import make_env
from timeit import default_timer as timer


# Arguments
map_name        = "zigzag_dists"        # map used for training
steps           = "1e4"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
checkpoint_freq = 100000                # Checkpoint save frequency
seed            = 123                   # Seed for pseudo random generators
domain_rand     = 1                     # Domain randomization (0 or 1)
checkpoint_cb   = False                 # Use checkpoints

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

# Create wrapped environment
env = make_env(map_name, log_dir, seed=seed, domain_rand=domain_rand, color_segment=False, FS=3, action_wrapper="heading")
env.reset()

# Create model
start = timer()

model = PPO(
        "CnnPolicy",
        env,
        learning_rate = float(LR),
        verbose = 1,
        tensorboard_log = tensorboard_log,
        seed = seed
        )

# Create checkpoint callback
if checkpoint_cb:
        checkpoint_callback = CheckpointCallback(
                save_freq = checkpoint_freq,
                save_path = f'../models/{map_name}/PPO/checkpoints/PPO_{steps}steps_lr{LR}_{color}_FS{FS}_DR{domain_rand}_heading',
                name_prefix = 'step_')
else:
        checkpoint_callback = None

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        tb_log_name = f"PPO_{steps}steps_lr{LR}_{color}_FS{FS}_DR{domain_rand}_heading"
        )

# Save trained model
model.save(f"../models/{map_name}/PPO/PPO_{steps}steps_lr{LR}_{color}_FS{FS}_DR{domain_rand}_heading")
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {int((end-start)/60)} mins\n")
