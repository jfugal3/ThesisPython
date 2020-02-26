from robosuite.environments.panda_door import PandaDoor
from mujoco_py import MjViewer
import gym
import numpy as np
import stable_baselines
# from stable_baselines.common.env_checker import check_env

class myPandaDoor(PandaDoor, gym.Env):
    def __init__(self):
        gym.Env.__init__(self)
        PandaDoor.__init__(self, has_renderer=False, use_camera_obs=False)
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
                0.0, # contact state
                -1.0, -1.0, -2*np.pi # door position
            ]),
            high=np.array([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
                4.0e-02, 0.0,
                0.5, 0.5,
                1.0,
                1.0, 1.0, 2*np.pi
            ]),
            dtype=np.float32
        )

    def reset(self):
        PandaDoor.reset(self)
        return self.unpack_obs()

    def close(self):
        pass

    def unpack_obs(self):
        obs = PandaDoor._get_observation(self)
        qpos = obs["joint_pos"]
        qvel = obs["joint_vel"]
        gripper_pos = obs["gripper_qpos"]
        gripper_vel = obs["gripper_qvel"]
        contact = [float(obs["contact-obs"])]
        obj_state = obs["object-state"]
        obs = np.concatenate([qpos, qvel, gripper_pos, gripper_vel, contact, obj_state])
        return obs

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self.unpack_obs()
        # print("object-state:{}".format(obj_state))
        # print("gripper_pos:{}".format(gripper_pos))
        # print("gripper_vel:{}".format(gripper_vel))
        reward = PandaDoor.reward(self, action)
        info = {}
        done = PandaDoor._check_terminated(self) or (self.timestep >= self.horizon) and not self.ignore_done
        return obs, reward, done, info

    def render(self):
        self.myViewer.render()


if __name__ == "__main__":
    import numpy as np
    env = myPandaDoor()
    # check_env(env)
    action_band = 10
    for i in range(1000):
        action = action_band * (np.random.rand(9) - 0.5)
        # action[7:] = [1.0,1.0]
        obs, reward, done, info = env.step(action)
        # print(obs)
        # print("\n\n\n\n\n\n\n")
        env.render()
