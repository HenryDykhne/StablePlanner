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
from PlannedHopperEnv import PlannedHopper
from PlannedReacherEnv import PlannedReacher
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

ORIGINAL = False
ENV = "Hopper"
PUNISH = True
STEPS = 3
RL_ALG = "PPO"
RUN_NO = 3
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
    env = TimeLimit(gym.make('CarRacing-v0', continuous = False), 1000)#solving = score > 900 #under construction
    NUM_EPISODES_FOR_EVAL = 20
    costMultiplier = 0
elif ENV == "CarRacing" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedCarRacing(steps = STEPS, punish = PUNISH, continuous = False), 1000))#too slow
    NUM_EPISODES_FOR_EVAL = 20
    costMultiplier = 0.1#try playing with this
elif ENV == "CartPole" and ORIGINAL:
    env = TimeLimit(gym.make('CartPole-v1'), 200)#solving = score > 199
    costMultiplier = 0
elif ENV == "CartPole" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedCartPole(steps = STEPS, punish = PUNISH), 200))
    costMultiplier = 1
elif ENV == "Hopper" and ORIGINAL:
    env = gym.make('Hopper-v3')#solving = not sure. maybe 2000? #timelimit enforced by env automatically 1000 timesteps
    boxSpace = True
    costMultiplier = 0
elif ENV == "Hopper" and not ORIGINAL:
    env = Monitor(PlannedHopper(steps = STEPS, punish = PUNISH))
    mujoco = True
    singleActionSize = 3
    boxSpace = True
    costMultiplier = 0.3
elif ENV == "Reacher" and ORIGINAL:
    env = gym.make('Reacher-v2')#solving = not sure.
    boxSpace = True
    costMultiplier = 0
elif ENV == "Reacher" and not ORIGINAL:
    env = Monitor(PlannedReacher(steps = STEPS, punish = PUNISH))
    mujoco = True
    singleActionSize = 2
    boxSpace = True
    costMultiplier = 0.3
env.reset()

model_path = f"{MODEL_DIRECTORY}/600000"
loadedModel = PPO.load(model_path, env=env)

NUM_EPISODES = 1
totalCost = 0
totalTimesteps = 0
totalReward = 0
for ep in range(NUM_EPISODES):
    print("Episode Number: " + str(ep))
    oldActionList = None
    obs = env.reset()
    done = False
    while not done:
        totalTimesteps += 1
        #env.render()
        action, _states = loadedModel.predict(obs)
        print(action)
        obs, rewards, done, info = env.step(action)
        # totalReward += rewards
        # ##calculating cost
        # actionList = action
        # action = action[0]
        # if oldActionList is not None:
        #     for i in range(len(oldActionList)-1):
        #         #print(actionList)
        #         #print(self.oldActionList[1:])
        #         if actionList[i] != oldActionList[1:][i]:
        #             totalCost += costMultiplier * (len(oldActionList) - i)/len(oldActionList)
        #             break
        # oldActionList = actionList
print("AverageRewardPerEpisode:" + str(totalReward/NUM_EPISODES))
print("AverageCostPerEpisode:" + str(totalCost/NUM_EPISODES))
print("AverageCostPerTimestep:" + str(totalCost/totalTimesteps))