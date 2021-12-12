import os
import numpy as np
from gym_duckietown.simulator import Simulator
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from utils.env import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env

algo            = "A2C"                 # name of RL algo
map_name        = "zigzag_dists"        # map used for training
steps           = "1e6"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
domain_rand     = 1                     # Domain randomization (0 or 1)
action_wrapper  = "heading"             # Action Wrapper to use ("heading" or "leftrightbraking")
n_eval_episodes = 10                    # Number of evaluation episodes
info            = "optimized"           # Model information (e.g. "optimized", "base", etc.)

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"

#Load trained model
#model_name = f"{algo}_{steps}steps_lr{LR}_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}"
model_name = f"{algo}_2e6steps_GrayS_FS3_DR1_heading_optimized"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_dir    = f"../models/{map_name}/{algo}/"

if algo == "A2C":
        model = A2C.load(save_dir + model_name, print_system_info=True)
elif algo == "PPO":
        model = PPO.load(save_dir + model_name, print_system_info=True)
else:
        print("Invalid algorithm.")

# Print model hyperparameters           -------- NOT WORKING NOW: NEED YAML CONFIG FOR HPARAMS!!
# model_hparams = model.get_parameters()
# print("\033[92m" + "Model hyperparameters:\n" + "\033[0m")
# for key, value in model_hparams.items():
#     print("\033[92m" + key + ' : ' + str(value) + "\033[0m")

with open(f'../results/{algo}_evaluation-log.csv', 'a') as csv_file:
        csv_file.write(f'Evaluation results for {algo} id: {info};\n\n')

eval_maps = ['zigzag_dists', 'small_loop', 'udem1']

print("\nEvaluating model...\n")

for map in eval_maps:
        # Create and wrap evaluation environment
        eval_env = make_env(map_name=map, log_dir=f"../logs/zigzag_dists/{algo}_log/eval")     # make it wrapped the same as "env" but with n_envs=1
        eval_env = make_vec_env(lambda: eval_env, n_envs=1, seed=12345)

        # Test the agent
        obs = eval_env.reset()
        rewards = []
        lengths = []

        rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        eval_env.close()
        del eval_env

        # Write logs
        with open(f'../results/{algo}_evaluation-log.csv', 'a') as csv_file:
                csv_file.write('Map;' + map + '\n')
                csv_file.write('Rewards;' + str(rewards) + '\n')
                csv_file.write('Min;' + str(min(rewards)) + '\n')
                csv_file.write('Max;' + str(max(rewards)) + '\n')
                csv_file.write('Mean;' + str(np.mean(rewards)) + '\n')
                csv_file.write('Stdev;' + str(np.mean(rewards)) + '\n')
                csv_file.write('Lengths;' + str(lengths) + '\n')
                csv_file.write('Min;' + str(min(lengths)) + '\n')
                csv_file.write('Max;' + str(max(lengths)) + '\n')
                csv_file.write('Mean;'+ str(np.mean(lengths)) + '\n')
                csv_file.write('Stdev;' + str(np.mean(lengths)) + '\n')
                csv_file.write('\n')

        # Print infos
        print("==============================================================================")
        print(f"Evaluation results for: {map}\n")
        print('Rewards:    ', rewards)
        print('Min reward: ', min(rewards))
        print('Max reward: ', max(rewards))
        print('Mean reward:', np.mean(rewards))
        print('Std reward: ', np.std(rewards))
        print('\nEpisode lengths:', lengths)
        print('Min ep length:  ', min(lengths))
        print('Max ep length:  ', max(lengths))
        print('Mean ep length: ', np.mean(lengths))
        print('Std ep length:  ', np.std(lengths))

del model
print("==============================================================================")
print("\nEvaluation ready. Logfile saved.\n")
