from robosuite.environments.panda_free_space_traj import PandaFreeSpaceTraj
from mujoco_py import MjViewer
from robosuite.utils import mjcf_utils
import gym
import numpy as np
import glfw
import os
from helperFun import grav_options
from helperFun import normalize
from helperFun import unnormalize_sym

class mySpec:
    def __init__(self):
        self.id = "my_panda_free_space_traj"

superclass = PandaFreeSpaceTraj
class myPandaFreeSpace1Goal(superclass, gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, grav_option, has_renderer=False, target_xyz=[0.6, -0.2, 1.4]):
        gym.Env.__init__(self)
        self.grav_option = grav_option
        self.num_via_points = 1
        self.via_points = np.array([[0] + target_xyz])
        superclass.__init__(self, has_renderer=has_renderer, has_offscreen_renderer=False, use_camera_obs=False)
        # self.spec = mySpec()
        # self.viewer.set_camera(camera_id=0)
        self.has_renderer = has_renderer
        # self.reward_range = 10000.0
        self.horizon = 1000
        self.distance_reward_weight = 30
        self.via_point_reward = 100
        # if self.has_renderer:
        #     self.myViewer = MjViewer(self.sim)
        self.spec = mySpec()
        if self.grav_option == grav_options["perfect_comp"]:
            self.sim.model.opt.gravity[:] = np.zeros(3)

        self.action_low = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0])
        self.action_high = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


        self.action_space = gym.spaces.Box(
            low=-np.ones(7),
            high=np.ones(7),
            dtype=np.float32
            )

        self.obs_low = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, # joint position
            -2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100, # joint velocity
        ])
        self.obs_high = np.array([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
            ])

        self.observation_space = gym.spaces.Box(
            low=np.zeros(14),
            high=np.ones(14),
            dtype=np.float32
        )
        self.horizon = 500

        self.pose_reward_weight = 10
        self.home_pose = self.obs_low[:7] + (self.obs_high[:7] - self.obs_low[:7]) / 2


    def _place_points(self):
        pass

    def reset(self):
        superclass.reset(self)
        # if self.has_renderer:
        #     glfw.destroy_window(self.myViewer.window)
        #     self.myViewer = MjViewer(self.sim)
        if self.grav_option == grav_options["perfect_comp"]:
            self.sim.model.opt.gravity[:] = np.zeros(3)
        return self.unpack_obs()

    def close(self):
        pass

    def unpack_obs(self):
        state = self.sim.get_state()
        q = state[1][:7]
        qd = state[2][:7]
        obs = np.concatenate([q, qd])#, gripper_pos, gripper_vel, obj_state])
        return normalize(obs, self.obs_low, self.obs_high)

    def _get_reward(self, action):
        """
        Return the reward obtained for a given action. Overall, reward increases as the robot
        checks via points in order.
        """
        reward = 0
        max_dist = 2.0
        ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
        dist = np.linalg.norm(ee_pos[:3] - self.via_points[self.next_idx][1:])
        # print(self.ee_pos[:3])

        # check if robot hit the next via point
        if dist < self.dist_threshold:
            self.sim.model.site_rgba[self.next_idx] = mjcf_utils.GREEN
            reward += self.via_point_reward

        reward += self.distance_reward_weight * (1 - np.tanh(5 * dist))  # was 10

        # qpos = self.sim.get_state()[1][:7]
        # reward += self.pose_reward_weight * (1 - np.tanh(np.linalg.norm(qpos - self.home_pose)))
        # reward -= reward * (np.abs(np.abs(action) - 1) < 0.0001) / len(self.action_low)
        # if np.any(np.abs(np.abs(action) - 1) < 0.0001):
        #     reward = 0
        # if apply_bound_penalty:
        #     qpos = self.sim.get_state()[1][:7]
        #     qmax = self.obs_high[:7]
        #     qmin = self.obs_low[:7]
        #     if np.any(qpos < qmin + 0.0001) or np.any(qpos > qmax - 0.0001):
        #         self.timestep = 2000
        #     reward -= np.sum(np.logical_or(qpos < qmin + 0.0001, qpos > qmax - 0.0001)) / 7.0 * reward
        # if np.any(np.abs())
        return reward

    def _get_done(self):
        return (self.timestep >= self.horizon)

    def _get_info(self):
        return {}

    def step(self, action):
        tau = unnormalize_sym(action, self.action_low, self.action_high)
        self.sim.data.ctrl[:] = np.concatenate((tau,[0,0]))

        self.timestep += 1
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.step()

        obs = self.unpack_obs()
        reward = self._get_reward(action)
        done = self._get_done()
        info = self._get_info()
        return obs, reward, done, info

    # def render(self, mode="human"):
    #     # superclass.render(self)
    #     if not self.has_renderer:
    #         print("Environment not in render mode!")
    #         return
    #     self.myViewer.render()


if __name__ == "__main__":
    import numpy as np
    env = myPandaFreeSpaceTraj()
    action_band = 10
    for i in range(1000):
        action = action_band * (np.random.rand(9) - 0.5)
        # action[7:] = [1.0,1.0]
        obs, reward, done, info = env.step(action)
        # print(obs)
        # print("\n\n\n\n\n\n\n")
        env.render()
