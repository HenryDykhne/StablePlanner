import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotArrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), mean-std, mean+std, color=color, alpha=0.3)

ORIGINAL = False
ENV = "LunarLander"
PUNISH = True
STEPS = 3
RL_ALG = "PPO"
RUN_NAME = ENV + '/' + ('punish/' if PUNISH else 'noPunish/') + str(STEPS) + 'steps'

DATA_DIRECTORY = "data/" + RUN_NAME

rewards = pickle.load(open(f"{DATA_DIRECTORY}/rewards.p", "rb" ))
costs = pickle.load(open(f"{DATA_DIRECTORY}/costs.p", "rb" ))

print(rewards)
print(costs)
colors = ['g','r','b']
plt.figure()
for color in colors:
    plotArrays(rewards, color, RUN_NAME)
plt.legend(loc='best')
plt.savefig("images/costWeighedReward.png")

# plt.figure()
# for (mode, color) in zip(modes, colors):
#     if mode == MOD_PUNISH:
#         print(np.asarray(rewards[mode]) + np.asarray(costs[mode]))
#         plotArrays(np.asarray(rewards[mode]) + np.asarray(costs[mode]), color, mode)
#     else:
#         plotArrays(rewards[mode], color, mode)
# plt.legend(loc='best')
# plt.savefig("images/costCorrectedReward.png")