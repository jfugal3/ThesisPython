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

if __name__ == "__main__":
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
        for i in range(1):
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
        history['obs'] = np.stack(history['obs'], axis=1)
        print(history['obs'].shape)
    except StringException as e:
        print(e.what())
