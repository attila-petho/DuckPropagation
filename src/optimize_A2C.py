import os
import torch
import datetime
from logging import log
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env

from utils.hyperparameters import sample_a2c_params, TrialEvalCallback
from utils.env import make_env
from stable_baselines3.common.vec_env import DummyVecEnv


def optimize_agent(trial):
    """ 
    Train the model and optimize
    """
    model_hparams = sample_a2c_params(trial)
    env = make_vec_env(lambda: make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/A2C_log/"), n_envs=4, seed=0)
    eval_env = make_vec_env(lambda: make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/A2C_log/eval"), n_envs=1, seed=1234)     # make it wrapped the same as "env" BUT WITH n_envs=1 !!!
    model = A2C('CnnPolicy', env, verbose=1, **model_hparams)
    
    eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=None,
            log_path="../logs/zigzag_dists/A2C_log/eval",
            n_eval_episodes=3,
            eval_freq=10000/4,      # have to divide with number of parallel envs
            deterministic=True
        )
    
    try:
        model.learn(50000, callback=eval_callback)                                    # TODO: should be 50k
        with torch.no_grad():                                                         #Context-manager that disables gradient calculation
            ep_rewards, ep_lengths = evaluate_policy(model, eval_env, n_eval_episodes=5, return_episode_rewards=True)
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
    #torch.cuda.empty_cache()       # only try this if nothing else works (very expensive)
    torch.cuda.synchronize()

    if is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    # Write trial logs to csv
    with open('../hyperparameters/A2C_optimization-log.csv', 'a') as csv_file:
        csv_file.write('Trial Number ' + str(trial.number) + ' hyperparameters' + ';\n')
        for key, value in trial.params.items():
            csv_file.write(key + ';' + str(value) + '\n')
        csv_file.write('Rewards;' + str(ep_rewards) + '\n')
        csv_file.write('Min;' + str(min(ep_rewards)) + '\n')
        csv_file.write('Max;' + str(max(ep_rewards)) + '\n')
        csv_file.write('Mean;' + str(np.mean(ep_rewards)) + '\n')
        csv_file.write('Lengths;' + str(ep_lengths) + '\n')
        csv_file.write('Min;' + str(min(ep_lengths)) + '\n')
        csv_file.write('Max;' + str(max(ep_lengths)) + '\n')
        csv_file.write('Mean;'+ str(np.mean(ep_lengths)) + '\n')
        csv_file.write('\n')
    
    return np.mean(ep_rewards) + np.mean(ep_lengths)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    time_now  = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
    study_name = f"A2C_optimize_hparams_study_1_{time_now}"
    sampler = TPESampler(n_startup_trials=0, seed=123)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=0)
    print("\nOptimizing agents...\n")
    print(f"Sampler: {sampler} - Pruner: {pruner}")

    with open('../hyperparameters/A2C_optimization-log.csv', 'w') as csv_file:
        csv_file.write('Study:;' + study_name + ';\n')

    study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction='maximize')
    try:
        study.optimize(optimize_agent, n_trials=100, gc_after_trial=True)     # if memory usage is too high use: gc_after_trial=True

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
    
    # save best hyperparams
    with open('../hyperparameters/A2C_optimization-log.csv', 'a') as csv_file:
        csv_file.write('\n')
        csv_file.write('Optimized A2C hyperparameters' + ';\n')
        csv_file.write('Trial number' + str(besttrial.number) + ';\n')
        for key, value in besttrial.params.items():
            csv_file.write(key + ';' + str(value) + '\n')
    print("\nLogfile saved.")
    