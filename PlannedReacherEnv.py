import numpy as np
from gym import utils
from mujoco_env import MujocoEnv


class PlannedReacher(MujocoEnv, utils.EzPickle):
    def __init__(self, steps: int = 3, punish: bool = False):
        self.singleActionSize = 2
        self.steps = steps
        self.punish = punish
        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, "reacher.xml", 2, self.steps)

    def step(self, a):
        actionList = a
        a = actionList[:-(len(actionList)-self.singleActionSize)]
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
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
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        observation = np.concatenate((self._get_obs(), np.zeros(self.steps*self.singleActionSize)))
        return observation

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )