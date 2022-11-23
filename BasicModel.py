import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO
from PlannedLanderPunishEnv import PlannedLunarLanderPunish, TimeLimit
from PlannedLanderPunishEnv5Steps import PlannedLunarLanderPunish5Steps
from PlannedLanderNoPunishEnv import PlannedLunarLanderNoPunish
from PlannedLanderNoPunishEnv5Steps import PlannedLunarLanderNoPunish5Steps
from PlannedLanderNoPunishEnvSingle import PlannedLunarLanderNoPunishSingle
from PlannedLanderNoPunishEnvSingleRegularObs import PlannedLunarLanderNoPunishSingleRegularObs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

TOTAL_TIMESTEPS = 500000
INTERVALS = 10
TIMESTEPS_BETWEEN_SAVES = int(TOTAL_TIMESTEPS / INTERVALS)

RUN_NAME = "PPO_Punish_3Steps_5"

LOG_DIRECTORY = "logs"
MODEL_DIRECTORY = "models/" + RUN_NAME
DATA_DIRECTORY = "data"

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

env = Monitor(TimeLimit(PlannedLunarLanderPunish(), 500))
#env = Monitor(TimeLimit(PlannedLunarLanderNoPunish(), 500))
#env = Monitor(TimeLimit(PlannedLunarLanderNoPunish5Steps(), 500))
#env = Monitor(TimeLimit(PlannedLunarLanderNoPunishSingleRegularObs(), 500))
#env = Monitor(TimeLimit(PlannedLunarLanderPunish5Steps(), 500))
#env = TimeLimit(gym.make('LunarLander-v2'), 500)
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)


iters = 0
while iters < INTERVALS:
    iters += 1
    print("iters: ", iters)
    model.learn(total_timesteps=TIMESTEPS_BETWEEN_SAVES, reset_num_timesteps=False, tb_log_name=RUN_NAME)
    model.save(f"{MODEL_DIRECTORY}/{TIMESTEPS_BETWEEN_SAVES*iters}")

# for ep in range(5):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
        


