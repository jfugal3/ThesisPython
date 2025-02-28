from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_controller import myPandaIKController
# from robosuite.utils.transform_utils import rotation_matrix
import gym
import numpy as np
import controllers
from helperFun import grav_options
from helperFun import eulerAnglesToRotationMatrix
from helperFun import unnormalize_sym
from helperFun import normalize

superclass = myPandaFreeSpace1Goal
class myPandaIKWrapper3D(superclass):
    def __init__(self, has_renderer=False, target_xyz=[0.6, -0.2, 1.4]):
        superclass.__init__(self, has_renderer=has_renderer, target_xyz=target_xyz, grav_option=grav_options["perfect_comp"])
        self.pandaIKController = myPandaIKController(robot_jpos_getter=self._get_qpos, robot_jvel_getter=self._get_qvel)
        self.action_low = np.array([-1.0, -1.0, 0.0])
        self.action_high = np.array([1.0, 1.0, 1.0])
        self.action_space = gym.spaces.Box(low=-np.ones(3), high=np.ones(3), dtype=np.float32)
        self.obs_low = self.action_low
        self.obs_high = self.action_high
        self.observation_space = gym.spaces.Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)
        self.rotation = eulerAnglesToRotationMatrix([np.pi, 0., np.pi/2])
    #     # self.control_timestep = 1.0

    def unpack_obs(self):
        xpos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')]) - np.array([0,0,1])
        return normalize(xpos, self.obs_low, self.obs_high)

    def _get_qpos(self):
        return self.sim.get_state()[1][0:7]

    def _get_qvel(self):
        return self.sim.get_state()[2][0:7]

    def step(self, action):
        xpos = unnormalize_sym(action, self.action_low, self.action_high)
        # dpos = 0.05 * action
        # xpos_command = np.clip(xpos + dpos, self.action_low, self.action_high)
        qgoal = self.pandaIKController.get_qpos(xpos, self.rotation)
        for i in range(int(self.control_timestep / self.model_timestep)):
            torque = np.concatenate((controllers.PDControl(q=self._get_qpos(), qd=self._get_qvel(), qgoal=qgoal),[0,0]))
            self.sim.data.ctrl[:] = torque
            self.sim.step()

        obs = self.unpack_obs()
        reward = self._get_reward(action, apply_bound_penalty=False)
        xpos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')]) - np.array([0,0,1])
        # reward -= np.sum(np.logical_or(np.abs(xpos - self.action_high) < 0.0001, np.abs(xpos - self.action_low) < 0.0001)) / 3.0 * reward
        reward -= np.sum(np.logical_or(xpos < self.action_low + 0.001, xpos > self.action_high - 0.001))
        done = self._get_done()
        info = self._get_info()
        self.timestep += 1
        reward -= reward * np.sum(np.abs(action) > 0.99) / 3
        return obs, reward, done, info
