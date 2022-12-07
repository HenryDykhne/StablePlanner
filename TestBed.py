import gym
import os
import numpy as np
import pickle
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO, TRPO
from PlannedLanderEnv import PlannedLunarLander, TimeLimit
from PlannedMountainCarEnv import PlannedMountainCar
from PlannedCarRacingEnv import PlannedCarRacing
from PlannedCartPoleEnv import PlannedCartPole
from PlannedHopperEnv import PlannedHopper
from PlannedReacherEnv import PlannedReacher
from PlannedHalfCheetahEnv import PlannedHalfCheetah
from PlannedInvertedDoublePendulumEnv import PlannedInvertedDoublePendulum
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def evaluateModel(model, env, costMultiplier, NUM_EPISODES = 80, mujoco = False, singleActionSize = None):
    totalCost = 0
    totalTimesteps = 0
    costWeighedReward = 0
    for ep in range(NUM_EPISODES):
        print("Episode Number: " + str(ep))
        oldActionList = None
        obs = env.reset()
        done = False
        while not done:
            totalTimesteps += 1
            #print(obs)
            action, _states = model.predict(obs)
            #print(action)
            obs, rewards, done, info = env.step(action)
            costWeighedReward += rewards
            ##calculating cost
            if not ORIGINAL:
                if mujoco:
                    actionList = action
                    action = actionList[:-(len(actionList)-singleActionSize)]
                    if oldActionList is not None:
                        for i in range(int((len(oldActionList)-singleActionSize)/singleActionSize)):
                            distance = 0
                            for k in range(singleActionSize):
                                distance += (actionList[i * singleActionSize + k] - oldActionList[singleActionSize:][i * singleActionSize + k]) ** 2
                            distance = distance ** (1/2)
                            partialCost = costMultiplier * distance * (len(oldActionList) - (i * singleActionSize))/len(oldActionList)
                            #print(partialCost)
                            totalCost -= partialCost
                        #print(oldActionList)
                        #print(actionList)
                    oldActionList = actionList
                else:
                    actionList = action
                    action = action[0]
                    if oldActionList is not None:
                        for i in range(len(oldActionList)-1):
                            if actionList[i] != oldActionList[1:][i]:
                                totalCost -= costMultiplier * (len(oldActionList) - i)/len(oldActionList)
                                break
                    oldActionList = actionList
    print('AvgCostWeighedRewardPerEpisode', costWeighedReward/NUM_EPISODES, 'AvgCostPerEpisode', totalCost/NUM_EPISODES, 'AvgCostPerTimestep', totalCost/totalTimesteps)
    return costWeighedReward/NUM_EPISODES, totalCost/NUM_EPISODES, totalCost/totalTimesteps

TOTAL_TIMESTEPS = 1200000
TIMESTEPS_BETWEEN_SAVES = 60000
INTERVALS = TOTAL_TIMESTEPS / TIMESTEPS_BETWEEN_SAVES
NUM_EPISODES_FOR_EVAL = 80

mujoco = False
singleActionSize = 3
boxSpace = False

ORIGINAL = False
ENV = "CartPole"
PUNISH = False
STEPS = 5
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
    env = Monitor(TimeLimit(PlannedReacher(steps = STEPS, punish = PUNISH), 50))
    mujoco = True
    singleActionSize = 2
    boxSpace = True
    costMultiplier = 1
elif ENV == "InvertedDoublePendulum" and ORIGINAL:
    env = gym.make('InvertedDoublePendulum-v2')#solving = not sure.
    boxSpace = True
    costMultiplier = 0
elif ENV == "InvertedDoublePendulum" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedInvertedDoublePendulum(steps = STEPS, punish = PUNISH), 1000))
    mujoco = True
    singleActionSize = 1
    boxSpace = True
    costMultiplier = 3
elif ENV == "HalfCheetah" and ORIGINAL:
    env = gym.make('HalfCheetah-v3')#solving = not sure.
    boxSpace = True
    costMultiplier = 0
elif ENV == "HalfCheetah" and not ORIGINAL:
    env = Monitor(TimeLimit(PlannedHalfCheetah(steps = STEPS, punish = PUNISH), 1000))
    mujoco = True
    singleActionSize = 6
    boxSpace = True
    costMultiplier = 1
env.reset()

rewards = []
costs = []
costsPerTimestep = []
for runNum in range(5):
    if 'PPO' == RL_ALG:
        model = PPO('MlpPolicy' if (ORIGINAL or boxSpace) else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
    elif 'A2C' == RL_ALG:
        model = A2C('MlpPolicy' if (ORIGINAL or boxSpace) else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)
    elif 'TRPO' == RL_ALG:
        model = TRPO('MlpPolicy' if (ORIGINAL or boxSpace) else "MultiInputPolicy", env, verbose=1, tensorboard_log=LOG_DIRECTORY)

    iters = 0
    rewards.append([])
    costs.append([])
    costsPerTimestep.append([])
    temp = "Run" + str(runNum)
    model.save(f"{MODEL_DIRECTORY}/{temp}/{TIMESTEPS_BETWEEN_SAVES*iters}") #first untrained model
    avgCostWeightedReward, avgCost, totalTimesteps = evaluateModel(model, env, costMultiplier, NUM_EPISODES = NUM_EPISODES_FOR_EVAL, mujoco = mujoco, singleActionSize = singleActionSize)
    rewards[runNum].append(avgCostWeightedReward)
    costs[runNum].append(avgCost)
    while iters < INTERVALS:
        iters += 1
        print("iters: ", iters)
        env.reset()
        model.learn(total_timesteps=TIMESTEPS_BETWEEN_SAVES, reset_num_timesteps=False, tb_log_name=RUN_NAME + "/Run" + str(runNum))
        model.save(f"{MODEL_DIRECTORY}/{temp}/{TIMESTEPS_BETWEEN_SAVES*iters}")
        avgCostWeightedReward, avgCost, avgCostPerTimestep = evaluateModel(model, env, costMultiplier, NUM_EPISODES = NUM_EPISODES_FOR_EVAL, mujoco = mujoco, singleActionSize = singleActionSize)
        rewards[runNum].append(avgCostWeightedReward)
        costs[runNum].append(avgCost)
        costsPerTimestep[runNum].append(avgCostPerTimestep)
        pickle.dump(rewards, open(f"{DATA_DIRECTORY}/rewards.p", "wb"))
        pickle.dump(costs, open(f"{DATA_DIRECTORY}/costs.p", "wb"))
        pickle.dump(costsPerTimestep, open(f"{DATA_DIRECTORY}/costsPerTimestep.p", "wb"))