import os
from torch import nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from utils.env import make_env
from utils.hyperparameters import linear_schedule
from timeit import default_timer as timer


# Arguments
map_name        = "zigzag_dists"        # map used for training
steps           = "2e6"                 # train for 2M steps
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
checkpoint_freq = 100000                # Checkpoint save frequency
seed            = 123                   # Seed for pseudo random generators
domain_rand     = 1                     # Domain randomization (0 or 1)
checkpoint_cb   = True                  # Use checkpoints

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"


model_hparams = {
        "n_steps": 32,
        "batch_size": 256,
        "learning_rate": linear_schedule(0.8764017708061861),
        "ent_coef": 6.569939601034458e-05,
        "clip_range": 0.2,
        "n_epochs": 5,
        "gae_lambda": 0.92,
        "max_grad_norm": 0.8,
        "vf_coef": 0.24250834235539484,
        "policy_kwargs": dict(
            activation_fn=nn.ReLU
        ),
    }

# Print model hyperparameters
print("\033[92m" + "Model hyperparameters:\n" + "\033[0m")
for key, value in model_hparams.items():
    print("\033[92m" + key + ' : ' + str(value) + "\033[0m")
if checkpoint_cb:
    print("\nCheckpoints saving is on.\n")
else:
   print("\nCheckpoints saving is off.\n")

# Create directories for logs
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_dir = f"../logs/{map_name}/PPO_log/"
os.makedirs(log_dir, exist_ok=True)
tensorboard_log = f"../tensorboard/{map_name}/"
os.makedirs(tensorboard_log, exist_ok=True)

# Create wrapped, vectorized environment
env = make_env(map_name, log_dir, seed=seed, domain_rand=domain_rand, color_segment=False, FS=3, action_wrapper=action_wrapper)
env = make_vec_env(lambda: env, n_envs=4, seed=0)
env.reset()

# Create model
start = timer()

model = PPO(
        "CnnPolicy",
        env,
        verbose = 1,
        tensorboard_log = tensorboard_log,
        seed = seed,
        **model_hparams
        )

# Create checkpoint callback
if checkpoint_cb:
        checkpoint_callback = CheckpointCallback(
                save_freq = checkpoint_freq,
                save_path = f'../models/{map_name}/PPO/checkpoints/PPO_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_optimized',
                name_prefix = 'step_')
else:
        checkpoint_callback = None

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        tb_log_name = f"PPO_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_optimized"
        )

# Save trained model
save_path = f"../models/{map_name}/PPO/PPO_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_optimized"
model.save(save_path)
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {int((end-start)/60)} mins\n")
print(f"Saved model to:\t{save_path}.zip\n\nEnjoy!\n")

del model, env
