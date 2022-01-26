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
from datetime import datetime


# Load configuration and initialize variables
configpath = os.path.join(ROOT_DIR, 'config', 'train_config.yml')
configs = load_config(configpath)
print('Seed: ', configs['common_config']['seed'], '\n')

color = "ColS" if configs['common_config']['color_segment'] else "GrayS"
activation = configs['common_config']['activation_fn']
activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
map_name = configs['common_config']['map_name']
steps = configs['common_config']['steps']
FS = configs['common_config']['FS']
domain_rand = configs['common_config']['domain_rand']
action_wrapper = configs['common_config']['action_wrapper']
reward_wrapper = configs['common_config']['reward_wrapper']
checkpoint_cb = configs['common_config']['checkpoint_cb']
seed = configs['common_config']['seed']
color_segment=configs['common_config']['color_segment']
n_envs = configs['common_config']['n_envs']
checkpoint_freq = configs['common_config']['checkpoint_freq']
lr_schedule = configs['common_config']['lr_schedule']
learning_rate = configs['common_config']['learning_rate']
n_steps = configs['common_config']['n_steps']
gae_lambda = configs['common_config']['gae_lambda']
ent_coef = configs['common_config']['ent_coef']
vf_coef = configs['common_config']['vf_coef']
max_grad_norm = configs['common_config']['max_grad_norm']
use_rms_prop = configs['a2c_config']['use_rms_prop']
normalize_advantage = configs['a2c_config']['normalize_advantage']
ID = configs['common_config']['ID']
comment = configs['common_config']['comment']


# Load model hyperparameters from config file
learning_rate = linear_schedule(learning_rate) if lr_schedule == 'linear' else learning_rate
model_hparams = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "normalize_advantage": normalize_advantage,
        "policy_kwargs": dict(
            activation_fn=activation_fn[activation]
        ),
    }

# Print model hyperparameters
print("\033[92m" + "Model hyperparameters:\n" + "\033[0m")
for key, value in model_hparams.items():        # TODO: this should iterate through a dict of config params, not the model_hparams
    print("\033[92m" + key + ' : ' + str(value) + "\033[0m")
if checkpoint_cb:
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
                seed=seed,
                domain_rand=domain_rand,
                color_segment=color_segment,
                FS=FS,
                action_wrapper=action_wrapper,
                reward_wrapper=reward_wrapper)
env = make_vec_env(lambda: env, n_envs=4, seed=0)
env.reset()

# Create model
start = timer()

model = A2C(
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
                save_path = f'../models/{map_name}/A2C/checkpoints/A2C_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_{ID}',
                name_prefix = 'step_')
else:
        checkpoint_callback = None

# Start training
model.learn(
        total_timesteps = int(float(steps)),
        callback = checkpoint_callback,
        tb_log_name = f"A2C_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_{ID}"
        )

# Save trained model
save_path = f"../models/{map_name}/A2C/A2C_{steps}steps_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}_{ID}"
model.save(save_path)
env.close()

end = timer()

del model, env

# Write log to csv
now = datetime.now()
finishtime = now.strftime("%Y/%m/%d_%H:%M:%S")
with open(f'../logs/A2C_train-log.csv', 'a') as csv_file:
        csv_log = (f'\n{finishtime};{ID};{steps};{lr_schedule};{str(learning_rate)};{action_wrapper};{reward_wrapper};'
                        f'{color_segment};{FS};{domain_rand};{map_name};{str(int((end-start)/60))};{comment};')
        csv_file.write(csv_log)

# Print training time
print(f"\nThe trained model is ready.\n\nElapsed Time: {int((end-start)/60)} mins\n")
print(f"Saved model to:\t{save_path}.zip\n\nEnjoy!\n")