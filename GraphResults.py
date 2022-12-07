import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotArrays(vars, color, label, maxY = np.inf, minY = -np.inf, numMeasurements = None, interval = 20000):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    if numMeasurements != None:
        mean = mean[0:numMeasurements+1]
        std = std[0:numMeasurements+1]
    plt.plot(np.array(range(len(mean)))*interval, mean, color=color, label=label)
    plt.fill_between(np.array(range(len(mean)))*interval, np.maximum(mean-std, minY), np.minimum(mean+std,maxY), color=color, alpha=0.3)

def getData(ORIGINAL = False, ENV = None, PUNISH = None, STEPS = None, RL_ALG = None):
    RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps/' + RL_ALG
    if ORIGINAL:
        RUN_NAME = ENV + '/original/' + RL_ALG
    DATA_DIRECTORY = "data/" + RUN_NAME
    rewards = pickle.load(open(f"{DATA_DIRECTORY}/rewards.p", "rb" ))
    costs = pickle.load(open(f"{DATA_DIRECTORY}/costs.p", "rb" ))
    if os.path.isfile(f"{DATA_DIRECTORY}/costsPerTimestep.p"):
        costsPerTimestep = pickle.load(open(f"{DATA_DIRECTORY}/costsPerTimestep.p", "rb" ))
    else:
        costsPerTimestep = None
    return rewards, costs, costsPerTimestep, RUN_NAME

colors = ['green','purple','blue', 'k', 'orange', 'red', 'c', 'y']

graphPath = 'images/LunarLander/punish/3steps'##These were run for 400000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", numMeasurements = 20)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", numMeasurements = 20)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 20)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", numMeasurements = 20)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", numMeasurements = 20)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 20)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, numMeasurements = 20)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 20)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/LunarLander/punish/5steps'##These were run for 600000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", numMeasurements = 30)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 30)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", numMeasurements = 30)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 30)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, numMeasurements = 30)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 30)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/LunarLanderStochastic/punish/3steps'##These were run for 1200000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLanderStochastic", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLanderStochastic", RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLanderStochastic", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", minY = 0, interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account


plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/CartPole/punish/3steps'##These were run for 100000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "CartPole", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 5)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "CartPole", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 5)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", minY = 0, numMeasurements = 5)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")

#################

graphPath = 'images/CartPole/punish/5steps'##These were run for 200000 steps 20000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "CartPole", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 10)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO")
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "CartPole", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)")
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 10)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = True, STEPS = 5, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0)
rewards, costs, costsPerTimestep, runName = getData(ENV = "CartPole", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[6], "PPO(NoPunish)", minY = 0, numMeasurements = 10)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")

################# This is the special graph to test if this architecture provides benifits
graphPath = 'images/LunarLander/comparison'
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO_3Steps_Punished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 4, RL_ALG = "PPO")
plotArrays(rewards, colors[1], "PPO_4Steps_Punished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[2], "PPO_5Steps_Punished")

rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 1, RL_ALG = "PPO")
plotArrays(rewards, colors[7], "PPO_1Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO_3Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 4, RL_ALG = "PPO")
plotArrays(rewards, colors[5], "PPO_4Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(rewards, colors[6], "PPO_5Steps_NotPunished")

rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO_Original")

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")

#--

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO_3Steps_Punished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 4, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "PPO_4Steps_Punished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = True, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "PPO_5Steps_Punished")


rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 1, RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[7], "PPO_1Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 3, RL_ALG = "PPO")#no need for cost correction here
plotArrays(np.asarray(rewards), colors[4], "PPO_3Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 4, RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[5], "PPO_4Steps_NotPunished")
rewards, costs, costsPerTimestep, runName = getData(ENV = "LunarLander", PUNISH = False, STEPS = 5, RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[6], "PPO_5Steps_NotPunished")

rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "LunarLander", RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[3], "PPO_Original")

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")
plt.close()

############

graphPath = 'images/Hopper/punish/3steps'##These were run for 1200000 steps 60000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "Hopper", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "Hopper", RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[3], "PPO(UnmodifiedEnv)", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", minY = 0, interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account


plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costsPerTimestep), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costsPerTimestep), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Timestep")
plt.savefig(graphPath+"/costPerTimestep.png")
plt.close()

#################

graphPath = 'images/Hopper/noPunish/3steps'##These were run for 1200000 steps 60000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "Hopper", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", interval = 60000)

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Hopper", PUNISH = False, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0, interval = 60000)


plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")
plt.close()

#################

graphPath = 'images/InvertedDoublePendulum/punish/3steps'##These were run for 1200000 steps 60000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "InvertedDoublePendulum", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, maxY = 10000, interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "InvertedDoublePendulum", RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[3], "PPO(UnmodifiedEnv)", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, maxY = 10000, interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", minY = 0, interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account


plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costsPerTimestep), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costsPerTimestep), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Timestep")
plt.savefig(graphPath+"/costPerTimestep.png")
plt.close()

############
graphPath = 'images/InvertedDoublePendulum/comparison'
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "InvertedDoublePendulum", RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[3], "PPO(UnmodifiedEnv)", minY = 0, maxY = 10000, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "InvertedDoublePendulum", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO_3Steps_NotPunished", interval = 60000)

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

############

graphPath = 'images/Reacher/punish/3steps'##These were run for 1200000 steps 60000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "Reacher", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "Reacher", RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[3], "PPO(UnmodifiedEnv)", maxY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", minY = 0, interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costsPerTimestep), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costsPerTimestep), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "Reacher", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Timestep")
plt.savefig(graphPath+"/costPerTimestep.png")
plt.close()

############

graphPath = 'images/HalfCheetah/punish/3steps'##These were run for 1200000 steps 60000 length intervals
if not os.path.exists(graphPath):
    os.makedirs(graphPath)

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[0], "PPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(rewards, colors[1], "A2C", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(rewards, colors[2], "TRPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "HalfCheetah", RL_ALG = "PPO")
plotArrays(rewards, colors[3], "PPO(UnmodifiedEnv)", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Reward Per Episode")
plt.savefig(graphPath+"/costWeighedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[0], "PPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[1], "A2C", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(np.asarray(rewards) - np.asarray(costs), colors[2], "TRPO", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ORIGINAL = True, ENV = "HalfCheetah", RL_ALG = "PPO")
plotArrays(np.asarray(rewards), colors[3], "PPO(UnmodifiedEnv)", interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(rewards, colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the rewards

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost Corrected Reward Per Episode")
plt.savefig(graphPath+"/costCorrectedReward.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costs), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costs), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costs), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Episode")
plt.savefig(graphPath+"/cost.png")
plt.close()

plt.figure()
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[0], "PPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "A2C")
plotArrays(-np.asarray(costsPerTimestep), colors[1], "A2C", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = True, STEPS = 3, RL_ALG = "TRPO")
plotArrays(-np.asarray(costsPerTimestep), colors[2], "TRPO", minY = 0, interval = 60000)
rewards, costs, costsPerTimestep, runName = getData(ENV = "HalfCheetah", PUNISH = False, STEPS = 3, RL_ALG = "PPO")
plotArrays(-np.asarray(costsPerTimestep), colors[4], "PPO(NoPunish)", interval = 60000)#for comparison, what the costs would have been if it is not taught to take them into account

plt.legend(loc='best')
plt.xlabel("Training Steps")
plt.ylabel("Average Cost of Replanning Per Timestep")
plt.savefig(graphPath+"/costPerTimestep.png")
plt.close()