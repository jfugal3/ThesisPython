from robosuite.environments.panda_lift import PandaLift
from mujoco_py import MjViewer
from robosuite.utils import mjcf_utils
import gym
import numpy as np
import glfw
import os
from helperFun import grav_options
from helperFun import normalize
from helperFun import unnormalize_sym
from helperFun import eulerAnglesToRotationMatrix
from my_panda_IK_controller import myPandaIKController
import controllers

class mySpec:
    def __init__(self):
        self.id = "my_panda_free_space_traj"

superclass = PandaLift
class myPandaLift(superclass, gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, grav_option, has_renderer=False):
        gym.Env.__init__(self)
        self.grav_option = grav_option
        superclass.__init__(self,
                            has_renderer=has_renderer,
                            has_offscreen_renderer=False,
                            use_camera_obs=False,
                            use_object_obs=True,
                            render_visual_mesh=has_renderer,
                            reward_shaping=True)
        self.has_renderer = has_renderer
        self.horizon = 1000
        self.distance_reward_weight = 30
        self.via_point_reward = 100
        self.spec = mySpec()


        if self.grav_option == grav_options["no_comp"]:
            self.action_low = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -20.0, -20.0])
            self.action_high = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 20.0, 20.0])

        if self.grav_option == grav_options["perfect_comp"]:
            self.action_low = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -20.0, -20.0])
            self.action_high = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 20.0, 20.0])
            self.sim.model.opt.gravity[:] = np.zeros(3)

        if self.grav_option == grav_options["ee_PD_cont"]:
            self.action_low = np.array([-1.0, -1.0, 0, -20.0, -20.0])
            self.action_high = np.array([1.0, 1.0, 1.0, 20.0, 20.0])
            self.sim.model.opt.gravity[:] = np.zeros(3)
            self.IK_controller = myPandaIKController(robot_jpos_getter=self._get_qpos, robot_jvel_getter=self._get_qvel)
            self.rotation = eulerAnglesToRotationMatrix([np.pi, 0., np.pi/2])

        if self.grav_option == grav_options["q_PD_cont"]:
            self.action_low = np.array([
                -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -20.0, -20.0# joint position
            ])
            self.action_high = np.array([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 20.0, 20.0
            ])
            self.sim.model.opt.gravity[:] = np.zeros(3)

        self.action_space = gym.spaces.Box(
            high=np.ones(len(self.action_low)),
            low=-np.ones(len(self.action_low)),
            dtype=np.float32
        )


        self.obs_low = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, # joint position
            -2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100, # joint velocity
            0.0, 0.0, 0.0, # cube pos
            -1.0, -1.0, -1.0, -1.0, # quaternion
            0.0, 0.0, 0.0, # vector between gripper and cube
        ])
        self.obs_high = np.array([
                2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
                2.0, 2.0, 2.0, # cube pos
                1.0, 1.0, 1.0, 1.0, # quaternion
                2.0, 2.0, 2.0, # vector between gripper and cube
        ])

        self.observation_space = gym.spaces.Box(
            low=np.zeros(len(self.obs_low)),
            high=np.ones(len(self.obs_low)),
            dtype=np.float32
        )

        self.n_control_steps = int(self.control_timestep / self.model_timestep)

    def _get_qpos(self):
        return self.sim.get_state()[1][0:7]

    def _get_qvel(self):
        return self.sim.get_state()[2][0:7]

    def reset(self):
        superclass.reset(self)
        if self.grav_option == grav_options["perfect_comp"] or self.grav_option == grav_options["ee_PD_cont"] or self.grav_option == grav_options["q_PD_cont"]:
            self.sim.model.opt.gravity[:] = np.zeros(3)
        return self.unpack_obs()


    def unpack_obs(self):
        state = self._get_observation()
        q = state['joint_pos']
        qd = state['joint_vel']
        obj_state = state['object-state']
        obs = np.concatenate([q, qd, obj_state])#, gripper_pos, gripper_vel, obj_state])
        return normalize(obs, self.obs_low, self.obs_high)


    def step(self, action):
        a = unnormalize_sym(action, self.action_low, self.action_high)
        robot_action = a[:7]
        gripper_action = a[7:]

        if self.grav_option == grav_options["no_comp"] or self.grav_option == grav_options["perfect_comp"]:
            controllers.env_torque_control(sim=self.sim, control_steps=self.n_control_steps, torque=robot_action, gripper_force=gripper_action)
        elif self.grav_option == grav_options["ee_PD_cont"]:
            robot_action = a[:3]
            gripper_action = a[3:]
            qpos = self.IK_controller.get_qpos(eef_pos=robot_action, rotation=self.rotation)
            controllers.env_PD_control(sim=self.sim, control_steps=self.n_control_steps, qpos=qpos, gripper_force=gripper_action)
        elif self.grav_option == grav_options["q_PD_cont"]:
            controllers.env_PD_control(sim=self.sim, control_steps=self.n_control_steps, qpos=robot_action, gripper_force=gripper_action)

        self.timestep += 1

        obs = self.unpack_obs()
        reward, done, info = self._post_action(action=None)
        return obs, reward, done, info
