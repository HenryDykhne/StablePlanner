import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotArrays(vars, color, label, maxY = np.inf, minY = -np.inf):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, minY), np.minimum(mean+std,maxY), color=color, alpha=0.3)

def getData(ENV, PUNISH, STEPS, RL_ALG):
    RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps/' + RL_ALG
    DATA_DIRECTORY = "data/" + RUN_NAME
    rewards = pickle.load(open(f"{DATA_DIRECTORY}/rewards.p", "rb" ))
    costs = pickle.load(open(f"{DATA_DIRECTORY}/costs.p", "rb" ))
    return rewards, costs, RUN_NAME

colors = ['g','r','b']

graphPath = 'images/LunarLander/punish/3steps'##These were run for 400000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)

plt.legend(loc='best')
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/LunarLander/punish/5steps'##These were run for 600000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)

#################

graphPath = 'images/LunarLanderStochastic/punish/3steps'##These were run for 1200000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)

plt.legend(loc='best')
plt.savefig(graphPath+"/cost.png")