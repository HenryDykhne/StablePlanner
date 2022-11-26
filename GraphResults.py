import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotArrays(vars, color, label, maxY = np.inf, minY = -np.inf):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, minY), np.minimum(mean+std,maxY), color=color, alpha=0.3)

def getData(ORIGINAL = False, ENV = None, PUNISH = None, STEPS = None, RL_ALG = None):
    RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps/' + RL_ALG
    if ORIGINAL:
        RUN_NAME = ENV + '/original/' + RL_ALG
    DATA_DIRECTORY = "data/" + RUN_NAME
    rewards = pickle.load(open(f"{DATA_DIRECTORY}/rewards.p", "rb" ))
    costs = pickle.load(open(f"{DATA_DIRECTORY}/costs.p", "rb" ))
    return rewards, costs, RUN_NAME

colors = ['green','purple','blue', 'k', 'orange', 'red', 'pink']

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
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, runName = getData(ORIGINAL = True, ENV = "LunarLanderStochastic", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "Baseline(PPO)")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, runName = getData(ORIGINAL = True, ENV = "LunarLanderStochastic", RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[3], "Baseline(PPO)")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)


plt.legend(loc='best')
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/CartPole/punish/3steps'##These were run for 100000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)

plt.legend(loc='best')
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/CartPole/punish/5steps'##These were run for 100000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)

################# This is the special graph to test if this architecture provides benifits
graphPath = 'images/LunarLander/comparison'##These were run for 100000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO_3Steps_Punished")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 4, RL_ALG = "PPO")
plotArrays(rewards, colors[1], "PPO_4Steps_Punished")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[2], "PPO_5Steps_Punished")

# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
# plotArrays(rewards, colors[4], "PPO_3Steps_NotPunished")
# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 4, RL_ALG = "PPO")
# plotArrays(rewards, colors[5], "PPO_4Steps_NotPunished")
# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
# plotArrays(rewards, colors[6], "PPO_5Steps_NotPunished")

# rewards, costs, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
# plotArrays(rewards, colors[3], "PPO_Original")

plt.legend(loc='best')
plt.savefig(graphPath+"/costWeighedReward.png")

#--

plt.figure()
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO_3Steps_Punished")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 4, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "PPO_4Steps_Punished")
rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "PPO_5Steps_Punished")

# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
# plotArrays(np.asarray(rewards) - np.asarray(costs), colors[4], "PPO_3Steps_NotPunished")
# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 4, RL_ALG = "PPO")
# plotArrays(np.asarray(rewards) - np.asarray(costs), colors[5], "PPO_4Steps_NotPunished")
# rewards, costs, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
# plotArrays(np.asarray(rewards) - np.asarray(costs), colors[6], "PPO_5Steps_NotPunished")

# rewards, costs, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
# plotArrays(np.asarray(rewards) - np.asarray(costs), colors[3], "PPO_Original")

plt.legend(loc='best')
plt.savefig(graphPath+"/costCorrectedReward.png")
