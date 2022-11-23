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

env = Monitor(TimeLimit(PlannedLunarLanderPunish(), 500))
#env_function = lambda: Monitor(TimeLimit(PlannedLunarLanderNoPunish(), 500))
#env = Monitor(TimeLimit(PlannedLunarLanderPunish5Steps(), 500))

#env = gym.make('LunarLander-v2')
env.reset()

MODEL_DIRECTORY = "models/PPO_Punish_3Steps_5"
model_path = f"{MODEL_DIRECTORY}/350000.zip"
loadedModel = PPO.load(model_path, env=env)

NUM_EPISODES = 100
totalCost = 0
totalTimesteps = 0
for ep in range(NUM_EPISODES):
    print("Episode Number: " + str(ep))
    oldActionList = None
    obs = env.reset()
    done = False
    while not done:
        totalTimesteps += 1
        #env.render()
        action, _states = loadedModel.predict(obs)
        #print(action)
        obs, rewards, done, info = env.step(action)
        ##calculating cost
        actionList = action
        action = action[0]
        if oldActionList is not None:
            for i in range(len(oldActionList)-1):
                #print(actionList)
                #print(self.oldActionList[1:])
                if actionList[i] != oldActionList[1:][i]:
                    totalCost += 1.5 * (len(oldActionList) - i)/len(oldActionList)
                    break
        oldActionList = actionList
print("AverageCostPerEpisode:" + str(totalCost/NUM_EPISODES))
print("AverageCostPerTimestep:" + str(totalCost/totalTimesteps))