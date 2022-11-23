import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from PlannedLanderPunishEnv import PlannedLunarLanderPunish, TimeLimit
from PlannedLanderNoPunishEnv import PlannedLunarLanderNoPunish
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt



TOTAL_TIMESTEPS = 400000
INTERVALS = 10
TIMESTEPS_BETWEEN_SAVES = int(TOTAL_TIMESTEPS / INTERVALS)

ORIGINAL = 'original'
MOD_NO_PUNISH = 'modNoPunish'
MOD_PUNISH = 'modPunish'

LOG_DIRECTORY = "logs"
MODEL_DIRECTORY = "models"
DATA_DIRECTORY = "data"

def plotArrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), mean-std, mean+std, color=color, alpha=0.3)

def getEnv(mode):
    if mode == 'original':
        return TimeLimit(gym.make('LunarLander-v2'), 500)
    elif mode == 'modNoPunish':
        return Monitor(TimeLimit(PlannedLunarLanderNoPunish(), 500))
    else:
        return Monitor(TimeLimit(PlannedLunarLanderPunish(), 500))

def evaluateModel(model, env, mode):
    NUM_EPISODES = 100
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
            if mode != ORIGINAL:
                actionList = action
                action = action[0]
                if oldActionList is not None:
                    for i in range(len(oldActionList)-1):
                        if actionList[i] != oldActionList[1:][i]:
                            totalCost -= 2 * (len(oldActionList) - i)/len(oldActionList)
                            break
                oldActionList = actionList
    return costWeighedReward/NUM_EPISODES, totalCost/NUM_EPISODES, totalTimesteps

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

modes = [ORIGINAL, MOD_NO_PUNISH, MOD_PUNISH]
runs = range(5)
rewards = {ORIGINAL:[], MOD_NO_PUNISH:[], MOD_PUNISH:[]}
costs = {ORIGINAL:[], MOD_NO_PUNISH:[], MOD_PUNISH:[]}
for run in runs:  
    for mode in modes:
        rewards[mode].append([])
        costs[mode].append([])
        env = getEnv(mode)
        
        if ORIGINAL == mode:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
        else:
            model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)

        iters = 0
        while iters < INTERVALS:
            env.reset()
            iters += 1
            #print("iters: ", iters)
            model.learn(total_timesteps=TIMESTEPS_BETWEEN_SAVES, reset_num_timesteps=False, tb_log_name="PPO:" + mode + str(run))
            model.save(f"{MODEL_DIRECTORY}/{mode}/{TIMESTEPS_BETWEEN_SAVES*iters}")
            avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(PPO.load(f"{MODEL_DIRECTORY}/{mode}/{TIMESTEPS_BETWEEN_SAVES*iters}", env=env), env, mode)
            print(rewards)
            rewards[mode][run].append(avgCostWeightedReward)
            costs[mode][run].append(avgCost)

pickle.dump(rewards, open(f"{DATA_DIRECTORY}/rewards.p", "wb"))
pickle.dump(costs, open(f"{DATA_DIRECTORY}/costs.p", "wb"))

rewards = pickle.load(open(f"{DATA_DIRECTORY}/rewards.p", "rb" ))
costs = pickle.load(open(f"{DATA_DIRECTORY}/costs.p", "rb" ))

print(rewards)
print(costs)
colors = ['g','r','b']
for (mode, color) in zip(modes, colors):
    plotArrays(rewards[mode], color, mode)
plt.legend(loc='best')
plt.savefig("images/costWeighedReward.png")

plt.figure()
for (mode, color) in zip(modes, colors):
    if mode == MOD_PUNISH:
        print(np.asarray(rewards[mode]) + np.asarray(costs[mode]))
        plotArrays(np.asarray(rewards[mode]) + np.asarray(costs[mode]), color, mode)
    else:
        plotArrays(rewards[mode], color, mode)
plt.legend(loc='best')
plt.savefig("images/costCorrectedReward.png")