import os
from logging import log
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env

from utils.hyperparameters import sample_ppo, TrialEvalCallback
from utils.env import make_env

# help: https://github.com/DLR-RM/rl-baselines3-zoo/blob/7f51ee5f1cc2e4ca06195be717621b7d1637b09a/utils/exp_manager.py#L155

# TODO: revise sampler and pruner!, return hyperparameters!


def optimize_agent(trial):
    """ 
    Train the model and optimize
    """
    model_hparams = sample_ppo(trial)
    env = make_vec_env(lambda: make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/PPO_log/"), n_envs=8, seed=0)
    eval_env = make_vec_env(lambda: make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/PPO_log/eval"), n_envs=8)
    model = PPO('CnnPolicy', env, verbose=1, **model_hparams)
    eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=None,
            log_path="../logs/zigzag_dists/PPO_log/eval",
            n_eval_episodes=3,
            eval_freq=10000,
            deterministic=True
        )
    try:
        model.learn(20000, callback=eval_callback)                                    # TODO: should be 50k
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3)
        env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("\n====================")
            print("Sampled hyperparams:")
            print(model_hparams)
            raise optuna.exceptions.TrialPruned()

    return mean_reward

if __name__ == "__main__":      # TODO: add sampler, pruner, return the hyperparams
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    study_name = "PPO_optimize_hparams_study_1"
    sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)                                # TODO: THESE ARE DEFINITELY NOT GOOD!
    pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)       # TODO: THESE ARE DEFINITELY NOT GOOD!
    print(f"Sampler: {sampler} - Pruner: {pruner}")
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    try:
        study.optimize(optimize_agent, n_trials=10)     # if memory usage is too high use: gc_after_trial=True

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')