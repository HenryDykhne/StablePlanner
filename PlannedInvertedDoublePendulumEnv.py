import numpy as np
from gym import utils
from mujoco_env import MujocoEnv


class PlannedInvertedDoublePendulum(MujocoEnv, utils.EzPickle):
    def __init__(self, steps: int = 3, punish: bool = False):
        self.singleActionSize = 1
        self.steps = steps
        self.punish = punish
        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        MujocoEnv.__init__(self, "inverted_double_pendulum.xml", 5, self.steps)
        utils.EzPickle.__init__(self)

    def step(self, action):
        actionList = action
        action = actionList[:-(len(actionList)-self.singleActionSize)]
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)

        if self.oldActionList is not None and self.punish:
            # print('(',len( self.oldActionList), '-', self.singleActionSize, ')/', self.singleActionSize)
            # print('yo', (len(self.oldActionList)-self.singleActionSize)/self.singleActionSize)
            for i in range(int((len(self.oldActionList)-self.singleActionSize)/self.singleActionSize)):
                distance = 0
                for k in range(self.singleActionSize):
                    distance += (actionList[i * self.singleActionSize + k] - self.oldActionList[self.singleActionSize:][i*self.singleActionSize + k]) ** 2
                distance = distance ** (1/2)
                r -= 3 * distance * (len(self.oldActionList) - (i * self.singleActionSize))/len(self.oldActionList)

        self.oldActionList = actionList

        ob = np.concatenate((ob, self.oldActionList))
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                np.clip(self.sim.data.qvel, -10, 10),
                np.clip(self.sim.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * 0.1,
        )
        self.oldActionList = np.zeros(self.singleActionSize*self.steps)
        observation = np.concatenate((self._get_obs(), np.zeros(self.steps*self.singleActionSize)))
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]