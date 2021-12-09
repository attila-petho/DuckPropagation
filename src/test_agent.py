import os
import time
#import pyglet
import numpy as np
from statistics import mean
from gym_duckietown.simulator import Simulator
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from utils.env import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env



algo            = "PPO"                 # name of RL algo
map_name        = "zigzag_dists"       # map used for training
steps           = "1e6"                 # train for 1M steps
LR              = "5e-4"                # Learning Rate: 0.0005
FS              = 3                     # Frames to stack
color_segment   = False                 # Use color segmentation or grayscale images
domain_rand     = 1                     # Domain randomization (0 or 1)
action_wrapper  = "heading"    # Action Wrapper to use ("heading" or "leftrightbraking")
maxsteps        = 0                   # number of steps to take in the test environment

color = None
if color_segment:
        color = "ColS"
else:
        color = "GrayS"
model_name  = f"{algo}_{steps}steps_lr{LR}_{color}_FS{FS}_DR{domain_rand}_{action_wrapper}"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_dir    = f"../models/{map_name}/{algo}/"

# Create env

# Wrap environment
# env = Monitor(env)
# env = ResizeFrame(env, 84)
# env = CropFrame(env, 24)
# if color_segment:                       # GrayScale and ColorSegment wrappers should not be used at the same time!
#         env = ColorSegmentFrame(env)
# else:
#         env = GrayScaleFrame(env)
# env = StackFrame(env, FS)
# if action_wrapper == "heading":         # Action wrappers ("heading" can be given a 'type' parameter)
#         env = Heading2WheelVelsWrapper(env)
# elif action_wrapper == "leftrightbraking":
#         env = LeftRightBraking2WheelVelsWrapper(env)
# else:
#         print("Invalid action wrapper. Using default actions.")
# env = DtRewardPosAngle(env)
# env = DtRewardVelocity(env)
# env = DummyVecEnv([lambda: env])
# env = VecTransposeImage(env)

eval_env = make_env(map_name="zigzag_dists", log_dir="../logs/zigzag_dists/PPO_log/eval")     # make it wrapped the same as "env" BUT WITH n_envs=1 !!!
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecTransposeImage(eval_env)

# Load trained model
if algo == "A2C":
        model = A2C.load(save_dir + model_name, print_system_info=True)
elif algo == "PPO":
        model = PPO.load(save_dir + model_name, print_system_info=True)
else:
        print("Invalid algorithm.")

# Test the agent
obs = eval_env.reset()
episode_rewards = []
episode_lengths = []
rewards = 0.0
lengths = 0

for step in range(maxsteps):
    action, _ = model.predict(obs, deterministic=True)
    print("ACTION: ", action)
    observation, reward, done, misc = eval_env.step(action)
    print("Reward: ", reward)
    rewards += reward
    lengths+=1
    eval_env.render()
    time.sleep(0.034)
    if done:
        episode_rewards.append(rewards.item())
        episode_lengths.append(lengths)
        rewards = 0.0
        lengths = 0
        eval_env.render()
        eval_env.reset()
        print("Done.")


rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=5, return_episode_rewards=True)

print("====================================================================")
print('Rewards: ', rewards, '\tLengths: ', episode_lengths)
print('Min reward: ', min(rewards), '\tMin ep length: ', min(episode_lengths))
print('Max reward: ', max(rewards), '\tMax ep length: ', max(episode_lengths))
print('Mean reward: ', np.mean(rewards), '\tMean ep length: ', np.mean(episode_lengths))
print("====================================================================")

#episode_rewards.append(rewards.item())
#print("\nAverage reward over %d steps: %.2f" % (maxsteps, (sum(episode_rewards)/len(episode_rewards))))
#print(f"episode_rewards: {episode_rewards}")
#episode_lengths.append(lengths)
#print("\nAverage episode length over %d steps: %.2f" % (maxsteps, (sum(episode_lengths)/len(episode_lengths))))
#print(f"episode_rewards: {episode_lengths}")

eval_env.close()

del model, eval_env
