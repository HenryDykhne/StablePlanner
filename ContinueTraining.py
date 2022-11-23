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

TOTAL_TIMESTEPS = 400000
INTERVALS = 10
TIMESTEPS_BETWEEN_SAVES = int(TOTAL_TIMESTEPS / INTERVALS)

RUN_NAME = "PPO_NoPunish_5Steps_4"

env_function = lambda: Monitor(PlannedLunarLanderNoPunish())
env = DummyVecEnv([env_function])

# model_path = f"{MODEL_DIRECTORY}/10000.zip"
# loadedModel = A2C.load(model_path, env=env)
# trainingData = []
# for ep in range(200):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = loadedModel.predict(obs)
#         trainingData.append({})
#         trainingData[-1]['stateCur'] = obs
#         trainingData[-1]['actionCur'] = action
#         obs, rewards, done, info = env.step(action)
#         trainingData[-1]['stateNext'] = obs
#         trainingData[-1]['reward'] = rewards
#         trainingData[-1]['done'] = obs
# # print(trainingData[2])
# # print('----')
# dataPath = f"{DATA_DIRECTORY}/trainingData.p"
# pickle.dump( trainingData, open(dataPath, "wb" ) )
# loadedTrainingData = pickle.load(open(dataPath, "rb" ) )
# print(loadedTrainingData[2]['actionCur'])