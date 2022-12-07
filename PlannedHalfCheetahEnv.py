import numpy as np
from gym import utils
from mujoco_env import MujocoEnv


class PlannedHalfCheetah(MujocoEnv, utils.EzPickle):
    
    def __init__(self, steps: int = 3, punish: bool = False):
        self.singleActionSize = 6
        self.steps = steps
        self.punish = punish
        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        MujocoEnv.__init__(self, "half_cheetah.xml", 5, self.steps)
        utils.EzPickle.__init__(self)

    def step(self, action):
        actionList = action
        action = actionList[:-(len(actionList)-self.singleActionSize)]
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False

        ##messing with rewards to penalize changing plans. since its continuous, it only cares about the distance between actions
        if self.oldActionList is not None and self.punish:
            # print('(',len( self.oldActionList), '-', self.singleActionSize, ')/', self.singleActionSize)
            # print('yo', (len(self.oldActionList)-self.singleActionSize)/self.singleActionSize)
            for i in range(int((len(self.oldActionList)-self.singleActionSize)/self.singleActionSize)):
                distance = 0
                for k in range(self.singleActionSize):
                    distance += (actionList[i * self.singleActionSize + k] - self.oldActionList[self.singleActionSize:][i*self.singleActionSize + k]) ** 2
                distance = distance ** (1/2)
                reward -= 1 * distance * (len(self.oldActionList) - (i * self.singleActionSize))/len(self.oldActionList)

        self.oldActionList = actionList
        ob = np.concatenate((ob, self.oldActionList))
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        observation = np.concatenate((self._get_obs(), np.zeros(self.steps*self.singleActionSize)))
        return observation

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5