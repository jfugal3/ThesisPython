import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
from UKThesis import controllers

control_freq = 100
sim_time = 5
control_range = 200 * np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1])
env = suite.make("PandaFreeSpaceTraj", has_renderer=True, use_camera_obs=False, ignore_done=True, controller='joint_torque', control_freq=control_freq, control_range=control_range)

obs = env.reset()
env.viewer.set_camera(camera_id=0)
q = obs['joint_pos']
qd = obs['joint_vel']
qgoal = obs['joint_pos'] + 0.5
print(obs)
# torque = controllers.PDControl(q, qd, qgoal)
# print(torque)
# action = np.append(torque, 0)
# # print(env.dof)
# # obs, reward, done, info = env.step(np.array([0,0,0,0,0,0,0,0]))
# # print('q: {}\nqdot: {}\ngrip_q: {}\ngrip_qdot: {}\n'.format(['joint_pos'], obs['joint_vel'], obs['gripper_qpos'], obs['gripper_qvel']))
# num_cycles = int(sim_time * control_freq)
# pos_history = np.zeros((7,num_cycles))
# for i in range(num_cycles):
#     # print("action:{}\n dof:{}".format(action, env.dof))
#     obs, reward, done, info = env.step(action)
#     q = obs['joint_pos']
#     qd = obs['joint_vel']
#     torque = controllers.PDControl(q, qd, qgoal)
#     # print(torque)
#     action = np.append(torque, 0)
#     env.render()
#     pos_history[:,i] = q
#
# for i in range(7):
#     joint = i + 1
#     plt.figure(joint)
#     t = np.arange(num_cycles)/control_freq
#     plt.plot(t, pos_history[i,:], t, qgoal[i] * np.ones(num_cycles))
#     plt.title("Joint {}".format(joint))
#
#
# plt.show()
