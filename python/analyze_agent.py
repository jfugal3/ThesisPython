from my_panda_free_space_traj import myPandaFreeSpaceTraj
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from stable_baselines import DQN, PPO2, A2C, ACKTR, DDPG, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from generatePatternedTrajectories import print_count
from helperFun import StringException
from helperFun import grav_options
import sys
import pprint
import matplotlib.pyplot as plt
import matplotlib

def bin(x, bin_num, low=-1, high=1):
    bin_edges = np.linspace(low,high, bin_num+1)
    out = np.zeros(bin_num)
    for val in x:
        for i in range(bin_num):
            if bin_edges[i] <= val <= bin_edges[i+1]:
                out[i] += 1
                continue

    centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2
    out = out / len(x)
    return centers, out

if __name__ == "__main__":
    matplotlib.rc('font', size=13)
    try:
        if len(sys.argv) < 3:
            raise StringException("Usage: training_logs/<path to agent model> <grav-option>")
        # env = myPandaFreeSpaceTraj(has_renderer=True)
        run_name = sys.argv[1]
        grav_option = sys.argv[2]
        if grav_option == "ee_PD_cont":
            env = myPandaIKWrapper3D(has_renderer=False)
        else:
            env = myPandaFreeSpace1Goal(has_renderer=False, grav_option=grav_options[grav_option])
        model = ACKTR.load("training_logs/" + run_name)


        # mean_reward, n_steps = evaluate_policy(model, env, 10)
        # print("avg reward:{}\nnumber of steps:{}".format(mean_reward, n_steps))
        ## Play Agent
        history = {'obs':[], 'action':[], 'reward':[]}
        pp = pprint.PrettyPrinter()
        for i in range(10):
            obs = env.reset()
            count = 0
            done = False
            print(i)
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                history['obs'].append(obs)
                history['action'].append(action)
                history['reward'].append(reward)
                # cum_reward += reward
                # env.render()
                # if count % 100:
                #     print("Action")
                #     pp.pprint(action)
                #     print("Joint State")
                #     pp.pprint(obs[:7])
        history['obs'] = np.vstack(history['obs'])
        history['action'] = np.vstack(history['action'])
        history['reward'] = np.vstack(history['reward'])
        print(history['action'].shape)
        for i in range(7):
            plt.figure(1)
            plt.subplot(2,4, i + 1)
            x,y = bin(history['action'][:,i], 10)
            plt.bar(x,y, width=0.2)
            plt.ylim([0,1])
            plt.title('Joint {}'.format(i+1))
            if i in [3,4,5,6]:
                plt.xlabel('Normalize Joint Torque')
            if i in [0,4]:
                plt.ylabel('Density')

            plt.figure(2)
            plt.subplot(2,4, i + 1)
            # plt.figure()
            x,y = bin(history['obs'][:,i], 10, low=-0.05, high=1.05)
            plt.bar(x,y, width=0.1)
            plt.title('Joint {}'.format(i+1))
            if i in [3,4,5,6]:
                plt.xlabel('Normalized Joint Angle')
            if i in [0,4]:
                plt.ylabel('Density')

        # plt.subplot(1,7,2)
        # x,y = bin(history['action'][:,1], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        #
        # plt.subplot(1,7,3)
        # x,y = bin(history['action'][:,2], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        #
        # plt.subplot(1,7,4)
        # x,y = bin(history['action'][:,3], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        #
        # plt.subplot(1,7,5)
        # x,y = bin(history['action'][:,4], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        #
        # plt.subplot(1,7,6)
        # x,y = bin(history['action'][:,5], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        #
        # plt.subplot(1,7,7)
        # x,y = bin(history['action'][:,6], 10)
        # plt.bar(x,y, width=0.2)
        # plt.ylim([0,1])
        # plt.subplot(1,7,4)
        # plt.title('Action Distribution')
        plt.show()
        # print(history['obs'].shape)
    except StringException as e:
        print(e.what())
