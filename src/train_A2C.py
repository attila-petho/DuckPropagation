import os
from torch import nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from utils.env import make_env
from utils.hyperparameters import linear_schedule
from utils.rootdir import ROOT_DIR
from utils.configloader import load_config
from timeit import default_timer as timer


# Load configuration and initialize variables
configpath = os.path.join(ROOT_DIR, 'config', 'train_config.yml')
configs = load_config(configpath)
print('Seed: ', configs['common_config']['seed'])

color = "ColS" if configs['common_config']['color_segment'] else "GrayS"
activation = configs['common_config']['activation_fn']
activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
map_name = configs['common_config']['map_name']
steps = configs['common_config']['steps']
FS = configs['common_config']['FS']
domain_rand = configs['common_config']['domain_rand']
action_wrapper=configs['common_config']['action_wrapper']


# Load model hyperparameters from config file
model_hparams = {
        "learning_rate": configs['common_config']['learning_rate'],
        "n_steps": configs['common_config']['n_steps'],
        "gae_lambda": configs['common_config']['gae_lambda'],
        "ent_coef": configs['common_config']['ent_coef'],
        "vf_coef": configs['common_config']['vf_coef'],
        "max_grad_norm": configs['common_config']['max_grad_norm'],
        "use_rms_prop": configs['a2c_config']['use_rms_prop'],
        "normalize_advantage": configs['common_config']['normalize_advantage'],
        "policy_kwargs": dict(
            activation_fn=activation_fn[activation]
        ),
    }

# Print model hyperparameters
print("\033[92m" + "\nModel hyperparameters:\n" + "\033[0m")
for key, value in model_hparams.items():
    print("\033[92m" + key + ' : ' + str(value) + "\033[0m")
if configs['common_config']['checkpoint_cb']:
    print("\nCheckpoints saving is on.\n")
else:
   print("\nCheckpoints saving is off.\n")

# Create directories for logs
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_dir = f"../logs/{map_name}/A2C_log/"
os.makedirs(log_dir, exist_ok=True)
tensorboard_log = f"../tensorboard/{map_name}/"
os.makedirs(tensorboard_log, exist_ok=True)

# Create wrapped, vectorized environment
env = make_env(map_name,
                log_dir,
                seed=configs['common_config']['seed'],
                domain_rand=domain_rand,
                color_segment=configs['common_config']['color_segment'],
                FS=FS,
                action_wrapper=action_wrapper)
env = make_vec_env(lambda: env, n_envs=configs['common_config']['n_envs'], seed=0)
env.reset()

# Create model
start = timer()

model = A2C(
        "CnnPolicy",
        env,
        verbose = 1,
        tensorboard_log = tensorboard_log,
        seed = configs['common_config']['seed'],
        **model_hparams
        )

# Create checkpoint callback
if configs['common_config']['checkpoint_cb']:
        checkpoint_callback = CheckpointCallback(
                                        save_freq = configs['common_config']['checkpoint_freq'],
                                        save_path = f'../models/{map_name}/PPO/checkpoints/PPO_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}',
                                        name_prefix = 'step_')
else:
        checkpoint_callback = None

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        tb_log_name = f"A2C_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_optimized"
        )

# Save trained model
save_path = f"../models/{map_name}/A2C/A2C_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_optimized"
model.save(save_path)
env.close()

# Print training time
end = timer()
print(f"\nThe trained model is ready.\n\nElapsed Time: {int((end-start)/60)} mins\n")
print(f"Saved model to:\t{save_path}.zip\n\nEnjoy!\n")

del model, env
