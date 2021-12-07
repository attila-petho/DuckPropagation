import os
from logging import log
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env

from utils.hyperparameters import sample_ppo, TrialEvalCallback
from utils.env import make_env


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
        model.learn(50000, callback=eval_callback)                                    # TODO: should be 50k
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

    is_pruned = eval_callback.is_pruned
    
    del model.env, eval_env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return mean_reward

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    study_name = "PPO_optimize_hparams_study_1"
    sampler = TPESampler(n_startup_trials=0, seed=123)
    pruner = MedianPruner(n_startup_trials=0, n_warmup_steps=0)
    print("\nOptimizing agents...\n")
    print(f"Sampler: {sampler} - Pruner: {pruner}")
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    try:
        study.optimize(optimize_agent, n_trials=100)     # if memory usage is too high use: gc_after_trial=True

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n\n======================================================================\n\n")
    print("Optimization is ready.\n\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("\nBest trial:")
    besttrial = study.best_trial

    print("  Value: ", besttrial.value)

    print("  Params: ")
    for key, value in besttrial.params.items():
        print("    {}: {}".format(key, value))
    
    # save hyperparams
    with open('PPO_optimization-log.csv', 'w') as csv_file:
        for key, value in besttrial.params.items():
            csv_file.write('Optimized PPO hyperparameters' + ';')
            csv_file.write(key + ';' + value)
    print("\nLogfile saved.")