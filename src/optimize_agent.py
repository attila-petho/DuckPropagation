import os
from logging import log
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env

from utils.hyperparameters import optimize_ppo
from utils.env import make_env

# help: https://github.com/DLR-RM/rl-baselines3-zoo/blob/7f51ee5f1cc2e4ca06195be717621b7d1637b09a/utils/exp_manager.py#L155

def optimize_agent(trial):
    """ 
    Train the model and optimize
    """
    model_params = optimize_ppo(trial)
    env = make_vec_env(lambda: make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/PPO_log/"), n_envs=2, seed=0)
    model = PPO('CnnPolicy', env, verbose=1, **model_params)
    model.learn(10000)
    eval_env = make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/PPO_log/")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward

if __name__ == "__main__":
    map_name = "zigzag_dists"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log_dir = f"../logs/{map_name}/PPO_log/"
    os.makedirs(log_dir, exist_ok=True)
    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(optimize_agent, n_trials=100) # what is that?
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')