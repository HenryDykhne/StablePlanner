import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO, TRPO
from PlannedLanderEnv import PlannedLunarLander, TimeLimit
from PlannedMountainCarEnv import PlannedMountainCar
#from PlannedCarRacingEnv import PlannedCarRacing
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def evaluateModel(model, env, costMultiplier):
    NUM_EPISODES = 80
    totalCost = 0
    totalTimesteps = 0
    costWeighedReward = 0
    for ep in range(NUM_EPISODES):
        #print("Episode Number: " + str(ep))
        oldActionList = None
        obs = env.reset()
        done = False
        while not done:
            totalTimesteps += 1
            #print(obs)
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            costWeighedReward += rewards
            ##calculating cost
            if not ORIGINAL:
                actionList = action
                action = action[0]
                if oldActionList is not None:
                    for i in range(len(oldActionList)-1):
                        if actionList[i] != oldActionList[1:][i]:
                            totalCost -= costMultiplier * (len(oldActionList) - i)/len(oldActionList)
                            break
                oldActionList = actionList
    return costWeighedReward/NUM_EPISODES, totalCost/NUM_EPISODES, totalTimesteps

TOTAL_TIMESTEPS = 1200000
TIMESTEPS_BETWEEN_SAVES = 20000
INTERVALS = TOTAL_TIMESTEPS / TIMESTEPS_BETWEEN_SAVES

ORIGINAL = False
ENV = "LunarLanderStochastic"
PUNISH = True
STEPS = 3
RL_ALG = "PPO"
RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps/' + RL_ALG

if ORIGINAL:
    RUN_NAME = ENV + '/original/' + RL_ALG

LOG_DIRECTORY = "logs"
MODEL_DIRECTORY = "models/" + RUN_NAME
DATA_DIRECTORY = "data/" + RUN_NAME

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

if ENV == "LunarLander" and ORIGINAL:
    env = TimeLimit(gym.make('LunarLander-v2'), 500) #solving = score > 200
    costMultiplier = 0
elif ENV == "LunarLander" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedLunarLander(steps = STEPS, punish = PUNISH), 500))
    costMultiplier = 2
elif ENV == "LunarLanderStochastic" and ORIGINAL:
    env = TimeLimit(gym.make('LunarLander-v2', enable_wind=True), 500)
    costMultiplier = 0
elif ENV == "LunarLanderStochastic" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedLunarLander(steps = STEPS, punish = PUNISH, enable_wind=True), 500))
    costMultiplier = 2
elif ENV == "MountainCar" and ORIGINAL:
    env = TimeLimit(gym.make('MountainCar-v0'), 200)
    costMultiplier = 0
elif ENV == "MountainCar" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedMountainCar(steps = STEPS, punish = PUNISH), 200))
    costMultiplier = 1
elif ENV == "CarRacing" and ORIGINAL:
    env = TimeLimit(gym.make('CarRacing-v2'), 2000)#solving = score > 900
    costMultiplier = 0
elif ENV == "CarRacing" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedCarRacing(steps = STEPS, punish = PUNISH), 2000))
    costMultiplier = 2
env.reset()

rewards = []
costs = []
for runNum in range(5):
    if 'PPO' == RL_ALG:
        model = PPO('MlpPolicy' if ORIGINAL else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
    elif 'A2C' == RL_ALG:
        model = A2C('MlpPolicy' if ORIGINAL else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
    elif 'TRPO' == RL_ALG:
        model = TRPO('MlpPolicy' if ORIGINAL else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)

    iters = 0
    rewards.append([])
    costs.append([])
    temp = "Run" + str(runNum)
    model.save(f"{MODEL_DIRECTORY}/{temp}/{TIMESTEPS_BETWEEN_SAVES*iters}") #first untrained model
    avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(model, env, costMultiplier)
    rewards[runNum].append(avgCostWeightedReward)
    costs[runNum].append(avgCost)
    while iters < INTERVALS:
        iters += 1
        print("iters: ", iters)
        env.reset()
        model.learn(total_timesteps=TIMESTEPS_BETWEEN_SAVES, reset_num_timesteps=False, tb_log_name=RUN_NAME + "/Run" + str(runNum))
        model.save(f"{MODEL_DIRECTORY}/{temp}/{TIMESTEPS_BETWEEN_SAVES*iters}")
        avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(model, env, costMultiplier)
        rewards[runNum].append(avgCostWeightedReward)
        costs[runNum].append(avgCost)

pickle.dump(rewards, open(f"{DATA_DIRECTORY}/rewards.p", "wb"))
pickle.dump(costs, open(f"{DATA_DIRECTORY}/costs.p", "wb"))


        


