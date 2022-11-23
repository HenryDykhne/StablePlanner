import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO, TRPO
from PlannedLanderEnv import PlannedLunarLander, TimeLimit
from PlannedLanderNoPunishEnvSingleRegularObs import PlannedLunarLanderNoPunishSingleRegularObs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def evaluateModel(model, env):
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
                            totalCost -= 2 * (len(oldActionList) - i)/len(oldActionList)
                            break
                oldActionList = actionList
    return costWeighedReward/NUM_EPISODES, totalCost/NUM_EPISODES, totalTimesteps

TOTAL_TIMESTEPS = 400000
TIMESTEPS_BETWEEN_SAVES = 20000
INTERVALS = TOTAL_TIMESTEPS / TIMESTEPS_BETWEEN_SAVES
TIME_LIMIT = 500

ORIGINAL = False
ENV = "LunarLander"
PUNISH = True
STEPS = 3
RL_ALG = "PPO"
RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps'

if ORIGINAL:
    RUN_NAME = ENV + '/original/'  + str(STEPS) + 'steps'

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
    env = TimeLimit(gym.make('LunarLander-v2'), TIME_LIMIT)
elif ENV == "LunarLander" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedLunarLander(steps = STEPS, punish = PUNISH), TIME_LIMIT))
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
    avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(model, env)
    rewards[runNum].append(avgCostWeightedReward)
    costs[runNum].append(avgCost)
    while iters < INTERVALS:
        iters += 1
        print("iters: ", iters)
        env.reset()
        model.learn(total_timesteps=TIMESTEPS_BETWEEN_SAVES, reset_num_timesteps=False, tb_log_name=RUN_NAME + "/Run" + str(runNum))
        model.save(f"{MODEL_DIRECTORY}/{temp}/{TIMESTEPS_BETWEEN_SAVES*iters}")
        avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(model, env)
        rewards[runNum].append(avgCostWeightedReward)
        costs[runNum].append(avgCost)

pickle.dump(rewards, open(f"{DATA_DIRECTORY}/rewards.p", "wb"))
pickle.dump(costs, open(f"{DATA_DIRECTORY}/costs.p", "wb"))


        


