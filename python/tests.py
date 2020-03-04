import randVelGen
import bounds
import helperFun
import generatePatternedTrajectories
from my_panda_free_space_traj import myPandaFreeSpaceTraj
import controllers
import data_calc
import ikpy_panda_kinematics
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import my_panda_free_space_1goal
import gym
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D

def test_generatePatternVel():
    gen = randVelGen.RandVelGenerator()
    test1 = gen.generatePatternVel()
    test2 = gen.generatePatternVel()
    gen.setJointDirections([-1,1,-1,1,-1,0,0])
    test3 = gen.generatePatternVel()
    test4 = gen.generatePatternVel()
    gen.setJointDirections([1,0,-1,-1,1,-1,1])
    test5 = gen.generatePatternVel()
    test6 = gen.generatePatternVel()
    print(np.around(test1,4))
    print(np.around(test2,4))
    print(np.around(test3,4))
    print(np.around(test4,4))
    print(np.around(test5,4))
    print(np.around(test6,4))

    passed = True
    for num in test1:
        if num > 0.0:
            passed = False
    print("Test1:", "Passed" if passed else "Failed")

    passed = True
    for num in test2:
        if num > 0.0:
            passed = False
    print("Test2:", "Passed" if passed else "Failed")

    passed = True
    test34ans = [1,-1,1,-1,1,-1,-1]
    for i in range(7):
        if np.sign(test3[i]) != np.sign(test34ans[i]):
            passed = False
    print("Test3:", "Passed" if passed else "Failed")

    passed = True
    for i in range(7):
        if np.sign(test4[i]) != np.sign(test34ans[i]):
            passed = False
    print("Test4:", "Passed" if passed else "Failed")

    passed = True
    test56ans = [-1,-1,1,1,-1,1,-1]
    for i in range(7):
        if np.sign(test5[i]) != np.sign(test56ans[i]):
            passed = False
    print("Test5:", "Passed" if passed else "Failed")

    passed = True
    for i in range(7):
        if np.sign(test6[i]) != np.sign(test56ans[i]):
            passed = False
    print("Test6:", "Passed" if passed else "Failed")

def test_getBoundViolations():
    ub = np.array(bounds.UPPER_BOUND)
    lb = np.array(bounds.LOWER_BOUND)
    test1 = np.zeros(7)
    test1[[1,3,5]] = ub[[1,3,5]] + 0.1
    test1[[2,6,0]] = lb[[2,6,0]] - 0.1
    test1[4] = ub[4] - 1
    test2 = np.zeros(7)
    test2[:] = ub[:] - 1
    test2[4] = lb[4] - 0.1

    temp = bounds.getBoundViolations(test1)
    ans1 = [-1,1,-1,1,0,1,-1]
    passed = True
    for i in range(7):
        if temp[i] != ans1[i]:
            passed = False
    print("Test1:", "Passed" if passed else "Failed")

    temp = bounds.getBoundViolations(test2)
    ans2 = [0,0,0,0,-1,0,0]
    passed = True
    for i in range(7):
        if temp[i] != ans2[i]:
            passed = False
    print("Test2:", "Passed" if passed else "Failed")

def test_tableBoundViolation():
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    test1 = np.array([-0.0031, -0.9718, -0.7269, -2.4357, -0.0157, 1.5763, 0.7303]) #False
    test2 = np.array([0.0264, -0.0772, 0.1924, -2.8478, -0.0273, 2.8339, 0.7566]) #True
    test3 = np.array([-1.4870, -1.7289, 1.6138, -1.9814, -0.9856, 1.9304, 0.9981]) #True
    test4 = np.array([-0.5250, -0.6410, 0.1893, -1.3827, -0.2573, 2.1356, 0.7116]) #False
    test5 = np.array([-0.0133, 0.9846, 0.0365, -1.5491, 2.8629, 0.7630, 0.6254]) #True

    qd = np.zeros(7)

    state = MjSimState(time=0,qpos=test1,qvel=qd,act=None,udd_state={})
    sim.set_state(state)
    sim.step()
    print("Test1:", "Passed" if not bounds.tableBoundViolation(sim) else "Failed")
    state = MjSimState(time=0,qpos=test2,qvel=qd,act=None,udd_state={})
    sim.set_state(state)
    sim.step()
    print("Test2:", "Passed" if bounds.tableBoundViolation(sim) else "Failed")
    state = MjSimState(time=0,qpos=test3,qvel=qd,act=None,udd_state={})
    sim.set_state(state)
    sim.step()
    print("Test3:", "Passed" if bounds.tableBoundViolation(sim) else "Failed")
    state = MjSimState(time=0,qpos=test4,qvel=qd,act=None,udd_state={})
    sim.set_state(state)
    sim.step()
    print("Test4:", "Passed" if not bounds.tableBoundViolation(sim) else "Failed")
    state = MjSimState(time=0,qpos=test5,qvel=qd,act=None,udd_state={})
    sim.set_state(state)
    sim.step()
    print("Test5:", "Passed" if bounds.tableBoundViolation(sim) else "Failed")


def test_outOfBounds():
    ub = np.array(bounds.UPPER_BOUND)
    lb = np.array(bounds.LOWER_BOUND)
    test1 = np.zeros(7)
    test1[[1,3,5]] = ub[[1,3,5]] + 0.1
    test1[[2,6,0]] = lb[[2,6,0]] - 0.1
    test1[4] = ub[4] - 1
    test2 = np.zeros(7)
    test2[:] = ub[:] - 1
    test2[4] = lb[4] - 0.1
    test3 = [-0.0031, -0.9718, -0.7269, -2.4357, -0.0157, 1.5763, 0.7303]
    test4 = [-0.5250, -0.6410, 0.1893, -1.3827, -0.2573, 2.1356, 0.7116]

    print("Test1:", "Passed" if bounds.outOfBounds(test1) else "Failed")
    print("Test2:", "Passed" if bounds.outOfBounds(test2) else "Failed")
    print("Test3:", "Passed" if not bounds.outOfBounds(test3) else "Failed")
    print("Test4:", "Passed" if not bounds.outOfBounds(test4) else "Failed")

def test_moving():
    test1 = np.zeros(7)
    test2 = np.array([0.1, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    test3 = np.array([0.01,0.02,0.0,0.0,0.0,0.0,0.0])
    print("Test1:", "Passed" if not helperFun.moving(test1) else "Failed")
    print("Test2:", "Passed" if helperFun.moving(test2) else "Failed")
    print("Test3:", "Passed" if not helperFun.moving(test3) else "Failed")

def test_stopped():
    test1 = np.zeros(7)
    test2 = np.array([0.1, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    test3 = np.array([0.01,0.02,0.0,0.0,0.0,0.0,0.0])
    test4 = np.array([0.0001,0.0002,0,0,0,0.000001])
    print("Test1:", "Passed" if helperFun.stopped(test1) else "Failed")
    print("Test2:", "Passed" if not helperFun.stopped(test2) else "Failed")
    print("Test3:", "Passed" if not helperFun.stopped(test3) else "Failed")
    print("Test4:", "Passed" if helperFun.stopped(test4) else "Failed")


def test_print_count():
    for i in range(1001):
        generatePatternedTrajectories.print_count(i)
        time.sleep(0.01)

def test_getUniqueFileName():
    print(helperFun.getUniqueFileName("test1"))
    print(helperFun.getUniqueFileName("test2"))
    print(helperFun.getUniqueFileName("test3"))
    print(helperFun.getUniqueFileName("test4"))


def test_getRandPosInBounds():
    test1 = bounds.getRandPosInBounds()
    test2 = bounds.getRandPosInBounds()
    test3 = bounds.getRandPosInBounds()
    test4 = bounds.getRandPosInBounds()
    print(np.around(test1,4))
    print(np.around(test2,4))
    print(np.around(test3,4))
    print(np.around(test4,4))
    print("Test1:", "Passed" if not bounds.outOfBounds(test1) else "Failed")
    print("Test2:", "Passed" if not bounds.outOfBounds(test2) else "Failed")
    print("Test3:", "Passed" if not bounds.outOfBounds(test3) else "Failed")
    print("Test4:", "Passed" if not bounds.outOfBounds(test4) else "Failed")


def test_PDControl():
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    timestep = generatePatternedTrajectories.TIMESTEP
    control_freq = 1/timestep
    total_time = 3
    num_cycles = int(total_time * control_freq)
    qd_init = np.zeros(7)
    plt.ion()
    LW = 1.0
    fig = plt.figure(figsize=(4,15))
    axes = []
    lines = []
    goals = []
    for i in range(7):
        axes.append(fig.add_subplot(7,1,i+1))
        lines.append(axes[i].plot([],[],'b-', lw=LW)[0])
        goals.append(axes[i].plot([],[],'r-', lw=LW)[0])
        axes[i].set_ylim([bounds.LOWER_BOUND[i], bounds.UPPER_BOUND[i]])
        axes[i].set_xlim([0,total_time])
        axes[i].set_ylabel("Angle{}(rad)".format(i), fontsize=8)
        axes[i].set_xlabel("Time(s)", fontsize=8)

    for test in range(5):
        q_init = bounds.getRandPosInBounds()
        q_goal = bounds.getRandPosInBounds()
        for g in range(7):
            goals[g].set_ydata(np.ones(num_cycles) * q_goal[g])
            goals[g].set_xdata(np.linspace(0,3,num_cycles))
        sim.set_state(MjSimState(time=0,qpos=q_init,qvel=qd_init,act=None,udd_state={}))
        sim.step()
        sim_time = 0
        for i in range(num_cycles):
            state = sim.get_state()
            q = state[1]
            qd = state[2]
            sim.data.ctrl[:] = controllers.PDControl(q=q,qd=qd,qgoal=q_goal)
            sim.step()
            viewer.render()
            if i % 70 == 0:
                for a in range(7):
                    lines[a].set_xdata(np.append(lines[a].get_xdata(), sim_time))
                    lines[a].set_ydata(np.append(lines[a].get_ydata(), q[a]))
                fig.canvas.draw()
                fig.canvas.flush_events()
            sim_time += timestep
        for i in range(7):
            lines[i].set_xdata([])
            lines[i].set_ydata([])
        time.sleep(1)


def test_basicVelControl():
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    timestep = generatePatternedTrajectories.TIMESTEP
    control_freq = 1/timestep
    total_time = 2
    num_cycles = int(total_time * control_freq)
    qd_init = np.zeros(7)
    plt.ion()
    LW = 1.0
    fig = plt.figure(figsize=(4,15))
    axes = []
    lines = []
    goals = []
    for i in range(7):
        axes.append(fig.add_subplot(7,1,i+1))
        lines.append(axes[i].plot([],[],'b-', lw=LW)[0])
        goals.append(axes[i].plot([],[],'r-', lw=LW)[0])
        axes[i].set_ylim([-1, 1])
        axes[i].set_xlim([0,total_time])
        axes[i].set_ylabel("Angle{}(rad)".format(i), fontsize=8)
        axes[i].set_xlabel("Time(s)", fontsize=8)

    for test in range(5):
        q_init = bounds.getRandPosInBounds()
        qd_goal = np.random.rand(7)
        for g in range(7):
            goals[g].set_ydata(np.ones(num_cycles) * qd_goal[g])
            goals[g].set_xdata(np.linspace(0,3,num_cycles))
        sim.set_state(MjSimState(time=0,qpos=q_init,qvel=qd_init,act=None,udd_state={}))
        sim.step()
        sim_time = 0
        for i in range(num_cycles):
            state = sim.get_state()
            q = state[1]
            qd = state[2]
            sim.data.ctrl[:] = controllers.basicVelControl(qd_des=qd_goal,qd_cur=qd)
            sim.step()
            viewer.render()
            if i % 35 == 0:
                for a in range(7):
                    lines[a].set_xdata(np.append(lines[a].get_xdata(), sim_time))
                    lines[a].set_ydata(np.append(lines[a].get_ydata(), qd[a]))
                fig.canvas.draw()
                fig.canvas.flush_events()
            sim_time += timestep
            if bounds.outOfBounds(q):
                break
        for i in range(7):
            lines[i].set_xdata([])
            lines[i].set_ydata([])
        time.sleep(1)


def test_dampingControl():
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    timestep = generatePatternedTrajectories.TIMESTEP
    control_freq = 1/timestep
    total_time = 2
    num_cycles = int(total_time * control_freq)
    plt.ion()
    LW = 1.0
    fig = plt.figure(figsize=(4,15))
    axes = []
    lines = []
    goals = []
    for i in range(7):
        axes.append(fig.add_subplot(7,1,i+1))
        lines.append(axes[i].plot([],[],'b-', lw=LW)[0])
        goals.append(axes[i].plot([],[],'r-', lw=LW)[0])
        axes[i].set_ylim([-1, 1])
        axes[i].set_xlim([0,total_time])
        axes[i].set_ylabel("Angle{}(rad)".format(i), fontsize=8)
        axes[i].set_xlabel("Time(s)", fontsize=8)

    for test in range(5):
        q_init = bounds.getRandPosInBounds()
        qd_goal = np.zeros(7)
        qd_init = np.random.rand(7)
        for g in range(7):
            goals[g].set_ydata(np.ones(num_cycles) * qd_goal[g])
            goals[g].set_xdata(np.linspace(0,3,num_cycles))
        sim.set_state(MjSimState(time=0,qpos=q_init,qvel=qd_init,act=None,udd_state={}))
        sim.step()
        sim_time = 0
        for i in range(num_cycles):
            state = sim.get_state()
            q = state[1]
            qd = state[2]
            sim.data.ctrl[:] = controllers.dampingControl(qd=qd)
            sim.step()
            viewer.render()
            if i % 35 == 0:
                for a in range(7):
                    lines[a].set_xdata(np.append(lines[a].get_xdata(), sim_time))
                    lines[a].set_ydata(np.append(lines[a].get_ydata(), qd[a]))
                fig.canvas.draw()
                fig.canvas.flush_events()
            sim_time += timestep
            if bounds.outOfBounds(q):
                break
        for i in range(7):
            lines[i].set_xdata([])
            lines[i].set_ydata([])
        time.sleep(1)


def test_command_zero_torque():
    env = myPandaFreeSpaceTraj()
    env.reset()

    for i in range(1000):
        env.step(np.zeros(9))
        env.render()


def test_inverse_kinematics():
    panda_kinematics = ikpy_panda_kinematics.panda_kinematics()
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    timestep = generatePatternedTrajectories.TIMESTEP
    control_freq = 1/timestep
    total_time = 3
    num_cycles = int(total_time * control_freq)
    qd_init = np.zeros(7)
    plt.ion()
    LW = 1.0
    fig = plt.figure(figsize=(4,15))
    axes = []
    lines = []
    goals = []
    min_vals = {'x': -0.5, 'y': -0.5, 'z': 0.2}
    max_vals = {'x': 0.5, 'y': 0.5, 'z': 0.7}
    ylabels = ["x(m)", "y(m)", "z(m)"]
    ylbounds = [min_vals['x'], min_vals['y'], min_vals['z']]
    yubounds = [max_vals['x'], max_vals['y'], max_vals['z']]
    for i in range(3):
        axes.append(fig.add_subplot(3,1,i+1))
        lines.append(axes[i].plot([],[],'b-', lw=LW)[0])
        goals.append(axes[i].plot([],[],'r-', lw=LW)[0])
        axes[i].set_ylim([ylbounds[i], yubounds[i]])
        axes[i].set_xlim([0,total_time])
        axes[i].set_ylabel(ylabels[i], fontsize=8)
        axes[i].set_xlabel("Time(s)", fontsize=8)

    for test in range(5):
        x_target = np.random.rand() * (max_vals['x'] - min_vals['x']) + min_vals['x']
        y_target = np.random.rand() * (max_vals['y'] - min_vals['y']) + min_vals['y']
        z_target = np.random.rand() * (max_vals['z'] - min_vals['z']) + min_vals['z']
        if\
            min_vals['x'] > x_target or max_vals['x'] < x_target or \
            min_vals['y'] > y_target or max_vals['y'] < y_target or \
            min_vals['z'] > z_target or max_vals['z'] < z_target:
            print("Error! 'xpos' out of range!")
            print("x = %.3f\ny = %.3f\nz = %.3f" % (x_target, y_target, z_target))

            print(max_vals['y'] - min_vals['y'])
            return
        # x_target, y_target, z_target = 0.088, -0.0, 0.926
        # roll = np.random.rand() * np.pi - np.pi/2
        # pitch = np.random.rand() * np.pi - np.pi/2
        # yaw = np.random.rand() * np.pi - np.pi/2
        roll = 0
        pitch = 0
        yaw = np.pi
        ee_goal = [x_target, y_target, z_target, roll, pitch, yaw]

        # temp = bounds.getRandPosInBounds()
        # ee_goal = panda_kinematics.forward_kinematics(temp)
        q_init = bounds.getRandPosInBounds()
        q_goal = panda_kinematics.inverse_kinematics(translation=ee_goal[0:3], rpy=ee_goal[3:6], init_qpos=q_init)

        for g in range(3):
            goals[g].set_ydata(np.ones(num_cycles) * ee_goal[g])
            goals[g].set_xdata(np.linspace(0,3,num_cycles))
        sim.set_state(MjSimState(time=0,qpos=q_init,qvel=qd_init,act=None,udd_state={}))
        sim.step()
        sim_time = 0
        for i in range(num_cycles):
            state = sim.get_state()
            q = state[1]
            qd = state[2]
            sim.data.ctrl[:] = controllers.PDControl(q=q,qd=qd,qgoal=q_goal)
            sim.step()
            viewer.render()
            if i % 70 == 0:
                xpos = panda_kinematics.forward_kinematics(q)
                for a in range(3):
                    lines[a].set_xdata(np.append(lines[a].get_xdata(), sim_time))
                    lines[a].set_ydata(np.append(lines[a].get_ydata(), xpos[a]))
                fig.canvas.draw()
                fig.canvas.flush_events()
                print("[q\t]:{}".format(np.around(q,3)))
                print("[qgoal\t]:{}".format(np.around(q_goal,3)))
                print("[qgoal2\t]:{}".format(np.around(panda_kinematics.inverse_kinematics(ee_goal[0:3], ee_goal[3:6]),3)))
                print("[EE\t]:{}".format(np.around(panda_kinematics.forward_kinematics(q),3)))
                print("[EEgoal\t]:{}".format(np.around(ee_goal,3)))
            sim_time += timestep
        # panda_kinematics.plot_stick_figure(q)
        for i in range(3):
            lines[i].set_xdata([])
            lines[i].set_ydata([])
        time.sleep(1)


def test_inverse_kinematics_no_plot():
    panda_kinematics = ikpy_panda_kinematics.panda_kinematics()

    xpos = panda_kinematics.forward_kinematics(np.zeros(7))
    theta = panda_kinematics.inverse_kinematics(xpos[0:3], xpos[3:6])
    print(np.around(xpos,4))
    print(np.around(panda_kinematics.forward_kinematics(theta),4))

    xpos = panda_kinematics.forward_kinematics(np.ones(7))
    theta = panda_kinematics.inverse_kinematics(xpos[0:3], xpos[3:6])
    print(np.around(xpos,4))
    print(np.around(panda_kinematics.forward_kinematics(theta),4))

    xpos = panda_kinematics.forward_kinematics(bounds.getRandPosInBounds())
    theta = panda_kinematics.inverse_kinematics(xpos[0:3], xpos[3:6])
    print(np.around(xpos,4))
    print(np.around(panda_kinematics.forward_kinematics(theta),4))


def test_get_rotation_get_orientation():
    phi = 0
    theta = 0
    psi = 0
    R = data_calc.get_rotation_matrix(phi,theta,psi)
    euler = data_calc.get_orientation(R)
    new_phi = euler[2]
    new_theta = euler[1]
    new_psi = euler[0]
    print(phi,new_phi)
    print(theta,new_theta)
    print(psi, new_psi)

    phi = 1
    theta = 1
    psi = 1
    R = data_calc.get_rotation_matrix(phi,theta,psi)
    euler = data_calc.get_orientation(R)
    new_phi = euler[2]
    new_theta = euler[1]
    new_psi = euler[0]
    print(phi,new_phi)
    print(theta,new_theta)
    print(psi, new_psi)

    phi = 1.265
    theta = 2.132
    psi = -1.459
    R = data_calc.get_rotation_matrix(phi,theta,psi)
    euler = data_calc.get_orientation(R)
    new_phi = euler[2]
    new_theta = euler[1]
    new_psi = euler[0]
    print(phi,new_phi)
    print(theta,new_theta)
    print(psi, new_psi)



def test_euler2rot_rot2euler():
    theta1_test1=np.zeros(3)
    R = ikpy_panda_kinematics.eulerAnglesToRotationMatrix(theta1_test1)
    theta1_ans1 = ikpy_panda_kinematics.rotationMatrixToEulerAngles(R)
    print(theta1_test1,"\n",theta1_ans1,"\n")

    theta1_test1=np.ones(3)
    R = ikpy_panda_kinematics.eulerAnglesToRotationMatrix(theta1_test1)
    theta1_ans1 = ikpy_panda_kinematics.rotationMatrixToEulerAngles(R)
    print(theta1_test1,"\n",theta1_ans1,"\n")

    theta1_test1=np.array([1.265, 2.132, -1.459])
    R = ikpy_panda_kinematics.eulerAnglesToRotationMatrix(theta1_test1)
    theta1_ans1 = ikpy_panda_kinematics.rotationMatrixToEulerAngles(R)
    print(theta1_test1,"\n",theta1_ans1,"\n")


def test_inverse_kinematics_pos():
    ik = ikpy_panda_kinematics.panda_kinematics()
    print("\nTest 1: First 3 elements of xpos should be the same.")
    pos = np.zeros(7)
    xpos = ik.forward_kinematics(pos)
    R1 = ik.euler_angles_to_rpy_rotation_matrix(xpos[3:6])
    print("xpos:", np.around(xpos,3))

    pos = ik.inverse_kinematics(xpos[0:3], xpos[3:6])
    print("pos: ", np.around(pos,3))

    xpos = ik.forward_kinematics(pos)
    R2 = ik.euler_angles_to_rpy_rotation_matrix(xpos[3:6])
    print("xpos:", np.around(xpos,3))

    print("\nThe 2 following matricies should be the same.")
    print(np.around(R1,3),"\n\n",np.around(R1,3))


    print("\nTest 2: First 3 elements of xpos should be the same.")
    pos = bounds.getRandPosInBounds()
    xpos = ik.forward_kinematics(pos)
    R1 = ik.euler_angles_to_rpy_rotation_matrix(xpos[3:6])
    print("xpos:", np.around(xpos,3))

    pos = ik.inverse_kinematics(xpos[0:3], xpos[3:6])
    print("pos: ", np.around(pos,3))

    xpos = ik.forward_kinematics(pos)
    R2 = ik.euler_angles_to_rpy_rotation_matrix(xpos[3:6])
    print("xpos:", np.around(xpos,3))

    print("\nThe 2 following matricies should be the same.")
    print(np.around(R1,3),"\n\n",np.around(R1,3))

def test_cycle_through_orientations():
    panda_kinematics = ikpy_panda_kinematics.panda_kinematics()
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    x_target, y_target, z_target = 0.2, 0.0, 0.5
    step1 = np.linspace(0,np.pi/2, 100)
    step2 = np.linspace(np.pi/2, -np.pi/2, 200)
    step3 = np.linspace(-np.pi/2, 0, 100)
    sweep = np.concatenate((step1, step2, step3))
    roll = 0
    pitch = 0
    yaw = np.pi


    ee_goal = [x_target, y_target, z_target, roll, pitch, yaw]
    qinit = panda_kinematics.inverse_kinematics(translation=ee_goal[0:3], rpy=ee_goal[3:6])

    sim.set_state(MjSimState(time=0,qpos=qinit,qvel=np.zeros(7),act=None,udd_state={}))
    sim.step()

    qgoal = qinit
    q = qinit
    qd = np.zeros(7)
    count = -1
    num_steps_per_change = 4
    for i in range(num_steps_per_change * 400):
        if i % num_steps_per_change == 0:
            count += 1
            ee_goal = [x_target, y_target, z_target, roll, pitch, sweep[count] + yaw]
            qgoal = panda_kinematics.inverse_kinematics(translation=ee_goal[0:3], rpy=ee_goal[3:6], init_qpos=q)
            R1 = panda_kinematics.euler_angles_to_rpy_rotation_matrix(rpy=[roll, pitch, sweep[count] + yaw])
            R2 = panda_kinematics.euler_angles_to_rpy_rotation_matrix(panda_kinematics.forward_kinematics(qgoal)[3:6])
            # R3 = sim.data.body_xmat[sim.model.body_name2id("right_hand")].reshape(3, 3)
            print("R1:\n", R1)
            print("R2:\n", R2)
            # print("R3:\n", R3)
            print("EE:", np.around(ee_goal, 3))
            print("q: ", np.around(qgoal, 3))

        state = sim.get_state()
        q = state[1]
        qd = state[2]
        sim.data.ctrl[:] = controllers.PDControl(q=q,qd=qd,qgoal=qgoal)
        sim.step()
        viewer.render()

    time.sleep(1)



def test_my_panda_free_space_1goal():
    env = my_panda_free_space_1goal.myPandaFreeSpace1Goal(has_renderer=True, grav_option=helperFun.grav_options["perfect_comp"])
    print(env.reset())
    for i in range(1000):
        obs, reward, done, info = env.step(np.random.uniform(low=-1.0, high=1.0))
        env.render()
        if i % 20 == 0:
            print(reward)


def test_grav_switch():
    gym.envs.registration.register(
        id="myPandaFreeSpace1Goal-v0",
        entry_point="my_panda_free_space_1goal:myPandaFreeSpace1Goal"
        )

    env = gym.make("myPandaFreeSpace1Goal-v0", grav_option=0, has_renderer=True)
    print("without gravity")
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()

    env.reset()
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()

    env.reset()
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()

    env = gym.make("myPandaFreeSpace1Goal-v0", grav_option=1, has_renderer=True)
    print("with gravity")
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()

    env.reset()
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()

    env.reset()
    for i in range(1000):
        env.step(np.zeros(7))
        env.render()


def test_myPandaIKWrapper3D():
    env = myPandaIKWrapper3D(has_renderer=True)
    # print(env.action_space)
    # goal1 = np.array([0.25, 0.5, 0.25])
    goal1 = np.array([0.6, -0.2, 0.4])
    for i in range(1000):
        obs, reward, done, info = env.step(goal1)
        # print(reward)
        env.render()


def test_normalize_unnormalize_sym():
    test1 = np.linspace(-3.12, 4.3, 10)
    test2 = np.linspace(-50.4, -2.1, 10)
    test3 = np.linspace(41.4, 60.5, 10)
    nom = np.linspace(-1, 1, 10)

    print("testing normalize_sym output ranges should be evenly distributed between -1 and +1.")
    print("Test 1", np.around(helperFun.normalize_sym(test1, -3.12, 4.3),2))
    print("Test 2", np.around(helperFun.normalize_sym(test2, -50.4, -2.1),2))
    print("Test 3", np.around(helperFun.normalize_sym(test3, 41.4, 60.5),2))

    print("testing unnormalize_sym")
    print("Test 1: range [-3.12, 4.3]\n", np.around(helperFun.unnormalize_sym(nom, -3.12, 4.3),2))
    print("Test 2: range [-50.4, -2.1]\n", np.around(helperFun.unnormalize_sym(nom, -50.4, -2.1),2))
    print("Test 3: range [41.4, 60.5]\n", np.around(helperFun.unnormalize_sym(nom, 41.4, 60.5),2))


def test_normalize_unnormalize():
    test1 = np.linspace(-3.12, 4.3, 10)
    test2 = np.linspace(-50.4, -2.1, 10)
    test3 = np.linspace(41.4, 60.5, 10)
    nom = np.linspace(0, 1, 10)

    print("testing normalize output ranges should be evenly distributed between 0 and +1.")
    print("Test 1", np.around(helperFun.normalize(test1, -3.12, 4.3),2))
    print("Test 2", np.around(helperFun.normalize(test2, -50.4, -2.1),2))
    print("Test 3", np.around(helperFun.normalize(test3, 41.4, 60.5),2))

    print("testing unnormalize")
    print("Test 1: range [-3.12, 4.3]\n", np.around(helperFun.unnormalize(nom, -3.12, 4.3),2))
    print("Test 2: range [-50.4, -2.1]\n", np.around(helperFun.unnormalize(nom, -50.4, -2.1),2))
    print("Test 3: range [41.4, 60.5]\n", np.around(helperFun.unnormalize(nom, 41.4, 60.5),2))



TEST_MAP = {
'test_generatePatternVel' : test_generatePatternVel,
'test_getBoundViolations' : test_getBoundViolations,
'test_tableBoundViolation' : test_tableBoundViolation,
'test_outOfBounds' : test_outOfBounds,
'test_moving' : test_moving,
'test_stopped' : test_stopped,
'test_print_count' : test_print_count,
'test_getUniqueFileName' : test_getUniqueFileName,
'test_getRandPosInBounds' : test_getRandPosInBounds,
'test_PDControl' : test_PDControl,
'test_basicVelControl' : test_basicVelControl,
'test_dampingControl' : test_dampingControl,
'test_command_zero_torque' : test_command_zero_torque,
'test_inverse_kinematics' : test_inverse_kinematics,
'test_inverse_kinematics_no_plot' : test_inverse_kinematics_no_plot,
'test_get_rotation_get_orientation' : test_get_rotation_get_orientation,
'test_euler2rot_rot2euler' : test_euler2rot_rot2euler,
'test_inverse_kinematics_pos' : test_inverse_kinematics_pos,
'test_cycle_through_orientations' : test_cycle_through_orientations,
'test_my_panda_free_space_1goal' : test_my_panda_free_space_1goal,
'test_grav_switch' : test_grav_switch,
'test_myPandaIKWrapper3D' : test_myPandaIKWrapper3D,
'test_normalize_unnormalize' : test_normalize_unnormalize,
'test_normalize_unnormalize_sym' : test_normalize_unnormalize_sym
}


def main():
    print()
    for test in sys.argv:
        if test == "tests.py":
            continue
        if test == "all":
            for key in TEST_MAP.keys():
                print(key)
                TEST_MAP[key]()
                print()
            break
        if test not in TEST_MAP.keys():
            print("'{}' does not exist in the current tests.".format(test))
            for key in TEST_MAP.keys():
                print(key)
        else:
            print(test)
            TEST_MAP[test]()
            print()


if __name__ == '__main__':
    main()
