from my_panda_free_space_traj import myPandaFreeSpaceTraj
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from stable_baselines import PPO2, ACKTR
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
import argparse
from env_creator import create_one_env

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
    parser = argparse.ArgumentParser("Anayze trained RL agent.")
    parser.add_argument('-env', required=True)
    parser.add_argument('-agent', required=True)
    parser.add_argument('-rn', required=True)
    args = parser.parse_args()
    # env = myPandaFreeSpaceTraj(has_renderer=True)
    training_dir = args.rn
    env = create_one_env(args.env)
    cum_reward = 0
    run_num = 1
    while cum_reward < 55000.0:
        print(run_num)
        cum_reward = 0
        model = {'PPO2': PPO2, 'ACKTR': ACKTR}[args.agent].load('training_logs/' + training_dir + '/run_' + str(run_num) + '_monitor_dirfinal_agent.pkl')
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            cum_reward += reward
        run_num += 1
        print(cum_reward)
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
        cum_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            history['obs'].append(obs)
            history['action'].append(action)
            history['reward'].append(reward)
            cum_reward += reward
            # env.render()
            # if count % 100:
            #     print("Action")
            #     pp.pprint(action)
            #     print("Joint State")
            #     pp.pprint(obs[:7])
        print(cum_reward)
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
            plt.xlabel('Normalized Joint Torque')
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
    task_num = 1
    if args.env.endswith('2'):
        task_num = 2
    if args.env.endswith('3'):
        task_num = 3
    # plt.figure(1)
    # plt.title('{} {} Normalized Joint Torque Histogram'.format({'PPO2': PPO2, 'ACKTR': ACKTR}[args.agent], task_num))
    plt.show()
