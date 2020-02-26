#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import controllers as ctr
import matplotlib.pyplot as plt
import data_calc

PI = np.pi

home = np.array([-0.0031, -0.9718, -0.7269, -2.4357, -0.0157, 1.5763, 0.7303])
next = np.array([-0.3861, -1.4530, 2.0671, -2.4727, -0.4629, 1.8478, 0.6507])

TIMESTEP = 0.002
control_freq = 1/TIMESTEP
sim_time = 3

model = load_model_from_path("robot.xml")
sim = MjSim(model)
step = 0
sim_state = sim.get_state()
print(sim_state)

# viewer = MjViewer(sim)
# num_cycles = int(sim_time * control_freq)
# pos_history = np.zeros((7,num_cycles))
# R3_history = pos_history.copy()
# for i in range(num_cycles):
#     sim_state = sim.get_state()
#     q = sim_state[1]
#     qd = sim_state[2]
#     torque = ctr.PDControl(q, qd, home)
#     # print(torque)
#     sim.data.ctrl[:] = torque
#     sim.step()
#     viewer.render()
#     # pos_history[:,i] = q
#     step += 1
#
# home = next
# for i in range(num_cycles):
#     sim_state = sim.get_state()
#     q = sim_state[1]
#     qd = sim_state[2]
#     torque = ctr.PDControl(q, qd, home)
#     # print(torque)
#     sim.data.ctrl[:] = torque
#     sim.step()
#     viewer.render()
#     pos_history[:,i] = q
#     R3_history[:,i] = data_calc.get_3D_data(sim)
#     step += 1
#
#
# # for i in range(7):
# #     joint = i + 1
# #     plt.figure(joint)
# #     t = np.arange(num_cycles)/control_freq
# #     plt.plot(t, pos_history[i,:], t, home[i] * np.ones(num_cycles))
# #     plt.title("Joint {}".format(joint))
#
# param_names = ["x","y","z","x_r","y_r","z_r","\psi"]
# for i in range(7,14):
#     plt.figure(i)
#     t = np.arange(num_cycles)/control_freq
#     plt.plot(t, R3_history[i-7,:])
#     plt.title("${}$".format(param_names[i-7]))
#
#
# plt.show()
