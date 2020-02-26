from robosuite.environments.panda_free_space_traj import PandaFreeSpaceTraj
# from robosuite.environments.panda import PandaEnv
from mujoco_py import MjViewer
from robosuite.utils import mjcf_utils
import gym
import numpy as np
import glfw
import os



class mySpec:
    def __init__(self):
        self.id = "my_panda_free_space_traj"

superclass = PandaFreeSpaceTraj
# superclass = PandaEnv
class myPandaFreeSpaceTraj(superclass, gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, has_renderer=False):
        gym.Env.__init__(self)
        superclass.__init__(self, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        # self.spec = mySpec()
        # self.viewer.set_camera(camera_id=0)
        self.has_renderer = has_renderer
        self.reward_range = 10000.0
        self.horizon = 1000
        self.distance_reward_weight = 10
        self.via_point_reward = 100
        if self.has_renderer:
            self.myViewer = MjViewer(self.sim)

        self.action_space = gym.spaces.Box(
            low=np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -20.0, -20.0]),
            high=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 20.0, 20.0]), dtype=np.float32
            )

        self.observation_space = gym.spaces.Box(
            low=np.array([
                -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, # joint position
                -2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100, # joint velocity
                0.0, -4.0e-02, # gripper position
                -0.5, -0.5, # gripper velocity
                 0.0, 0.5, -0.15, 1.4,  # goal state
                 0.0, 0.5, 0.15, 1.4,
                 0.0, 0.5, 0.15, 1.2,
                 0.0, 0.5, -0.15, 1.2
            ]),
            high=np.array([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
                4.0e-02, 0.0,
                0.5, 0.5,
                 1.0, 0.5, -0.15, 1.4,  # goal state
                 1.0, 0.5, 0.15, 1.4,
                 1.0, 0.5, 0.15, 1.2,
                 1.0, 0.5, -0.15, 1.2
            ]),
            dtype=np.float32
        )

    def reset(self):
        superclass.reset(self)
        if self.has_renderer:
            glfw.destroy_window(self.myViewer.window)
            self.myViewer = MjViewer(self.sim)
        return self.unpack_obs()

    def close(self):
        pass

    def unpack_obs(self):
        obs = superclass._get_observation(self)
        # print(obs)
        qpos = obs["joint_pos"]
        qvel = obs["joint_vel"]
        gripper_pos = obs["gripper_qpos"]
        gripper_vel = obs["gripper_qvel"]
        obj_state = obs["object-state"]
        obs = np.concatenate([qpos, qvel, gripper_pos, gripper_vel, obj_state])
        return obs

    def reward(self, action):
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
        if self.finished_time is None and dist < self.dist_threshold:
            self.sim.model.site_rgba[self.next_idx] = mjcf_utils.GREEN
            self.via_points[self.next_idx][0] = 1  # mark as visited
            self.next_idx += 1
            reward += self.via_point_reward

            # if there are still via points to go
            if self.next_idx != self.num_via_points:
                # color next target red
                self.sim.model.site_rgba[self.next_idx] = mjcf_utils.RED

        # reward for remaining distance
        # else:
        #     # if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
        #     if not self.use_delta_distance_reward:
        #         reward += self.distance_reward_weight * (1 - np.tanh(5 * dist))  # was 10
        #         # reward += self.distance_reward_weight * (max_dist - dist) / max_dist
        #     else:
        #         prev_dist = np.linalg.norm(self.prev_ee_pos[:3] - self.via_points[self.next_idx][1:])
        #         reward += self.distance_reward_weight * (prev_dist - dist)
        #         reward -= self.distance_penalty_weight * np.tanh(10 * dist)

        reward += self.distance_reward_weight * (self.next_idx + (1 - np.tanh(5 * dist))
        # What we want is to reach the points fast
        # We add a reward that is proportional to the number of points crossed already
        # reward += self.next_idx * self.acc_vp_reward_mult

        # penalize for taking another timestep
        # (e.g. 0.001 per timestep, for a total of 4096 timesteps means a penalty of 40.96)
        # reward -= self.timestep_penalty
        # Penalize time in episode
        # reward -= 30
        # penalize for jerkiness
        # reward -= self.energy_penalty * np.sum(np.abs(self.joint_torques))
        # reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
        # reward -= self.action_delta_penalty * np.mean(abs(self._compute_a_delta()[:3]))

        return reward

    def step(self, action):
        # action += np.random.normal(loc=0,scale=0.1,size=9)
        self.sim.data.ctrl[:] = action
        obs = self.unpack_obs()

        reward = self.reward(action)
        info = {}
        self.timestep += 1
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.step()

        done = superclass._check_success(self) or (self.timestep >= self.horizon) and not self.ignore_done
        return obs, reward, done, info

    def render(self, mode="human"):
        # superclass.render(self)
        if not self.has_renderer:
            print("Environment not in render mode!")
            return
        self.myViewer.render()


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
