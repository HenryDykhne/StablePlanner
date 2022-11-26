import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO, TRPO
from PlannedLanderEnv import PlannedLunarLander, TimeLimit
from LunarLander import LunarLander
from PlannedMountainCarEnv import PlannedMountainCar
from PlannedCarRacingEnv import PlannedCarRacing
from PlannedCartPoleEnv import PlannedCartPole
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

ORIGINAL = False
ENV = "CartPole"
PUNISH = True
STEPS = 5
RL_ALG = "PPO"
RUN_NO = 0
RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps/' + RL_ALG + '/Run' + str(RUN_NO)

if ORIGINAL:
    RUN_NAME = ENV + '/original/' + RL_ALG + '/Run' + str(RUN_NO) 

MODEL_DIRECTORY = "models/" + RUN_NAME

if ENV == "LunarLander" and ORIGINAL:
    env = TimeLimit(gym.make('LunarLander-v2'), 500) #solving = score > 200
    costMultiplier = 0
elif ENV == "LunarLander" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedLunarLander(steps = STEPS, punish = PUNISH), 500))
    costMultiplier = 2
elif ENV == "LunarLanderStochastic" and ORIGINAL:
    env = Monitor(TimeLimit(LunarLander(enable_wind=True), 500))
    costMultiplier = 0
elif ENV == "LunarLanderStochastic" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedLunarLander(steps = STEPS, punish = PUNISH, enable_wind=True), 500))
    costMultiplier = 2
elif ENV == "MountainCar" and ORIGINAL: #unsolvable by stock PPO
    env = TimeLimit(gym.make('MountainCar-v0'), 200)
    costMultiplier = 0
elif ENV == "MountainCar" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedMountainCar(steps = STEPS, punish = PUNISH), 200))
    costMultiplier = 1
elif ENV == "CarRacing" and ORIGINAL:
    env = TimeLimit(gym.make('CarRacing-v0', continuous = False), 2000)#solving = score > 900 #under construction
    costMultiplier = 0
elif ENV == "CarRacing" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedCarRacing(steps = STEPS, punish = PUNISH), 2000))
    costMultiplier = 2
elif ENV == "CartPole" and ORIGINAL:
    env = TimeLimit(gym.make('CartPole-v1', continuous = False), 200)#solving = score > 199
    costMultiplier = 0
elif ENV == "CartPole" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedCartPole(steps = STEPS, punish = PUNISH), 200))
    costMultiplier = 1
env.reset()

model_path = f"{MODEL_DIRECTORY}/180000.zip"
loadedModel = PPO.load(model_path, env=env)

NUM_EPISODES = 80
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