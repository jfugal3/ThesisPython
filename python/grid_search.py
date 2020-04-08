import warnings
warnings.simplefilter("ignore")
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
import my_panda_free_space_traj
from stable_baselines import PPO2, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import stable_baselines.common.cmd_util
from stable_baselines import logger
from helperFun import StringException
from helperFun import grav_options
from env_creator import create_env
from env_creator import ENVS
import argparse

import gym
import os
import numpy as np
from datetime import datetime
import pandas as pd
import sys
import gc

best_mean_reward, n_steps = -np.inf, 0
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, env, run_dir, checkpoint_dir
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(run_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                _locals['self'].save(run_dir + '/best_model.pkl')
    # Returning False will stop training early
    return True

def hyper_file_name(hyper_map):
    name = ""
    for key in sorted(hyper_map.keys()):
        name += key + "-" + str(hyper_map[key]) + "__"
    name = name[:len(name) - 2]
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grid Search")
    parser.add_argument("-ld", "--log_dir", required=True)
    parser.add_argument("-hd", "--hyperparamfile", required=True)
    parser.add_argument("-env", required=True, choices=['1goal_no_comp', '1goal_perfect_comp'])
    parser.add_argument("-n_envs", type=int, required=False, default=4)
    parser.add_argument("-ts", type=int, required=True)
    parser.add_argument("-agent", required=True, choices=['ACKTR', "PPO2"])
    args = parser.parse_args()
    AGENTS_MAP = {'ACKTR': ACKTR, 'PPO2':PPO2}

    top_log_dir = args.log_dir
    hyperparams_list = np.load(args.hyperparamfile, allow_pickle=True)
    env_name = args.env
    n_envs = args.n_envs
    timesteps = args.ts
    RLAgent = AGENTS_MAP[args.agent]

    start_time = datetime.now()

    top_log_dir = os.path.join("training_logs", top_log_dir)
    os.makedirs(top_log_dir, exist_ok=True)
    test_num = 1
    for hyperparams in hyperparams_list:
        hyperparam_log_dir = os.path.join(top_log_dir, hyper_file_name(hyperparams))
        os.makedirs(hyperparam_log_dir, exist_ok=True)
        print("Beginning test", test_num, "of", len(hyperparams_list))
        begin_perm_time = datetime.now()
        for i in range(5,10):
            run_dir = os.path.join(hyperparam_log_dir, "run_" + str(i) + "_monitor_dir")
            hyperparamfilename = os.path.join(run_dir, "hyperparams.txt")
            if os.path.exists(hyperparamfilename):
                continue
            os.makedirs(run_dir, exist_ok=True)
            checkpoint_dir = os.path.join(run_dir, "model_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.configure(run_dir)

            env = create_env(n_envs=n_envs, env_name=env_name, log_dir=run_dir)
            model = RLAgent('MlpPolicy', env, verbose=0, **hyperparams).learn(total_timesteps=timesteps, callback=callback)
            model.save(run_dir + "final_agent.pkl")
            del model
            del env
            gc.collect()
            hyperparamfile = open(hyperparamfilename, 'w')
            hyperparamfile.write(str(hyperparams))
            hyperparamfile.write("\nn_envs = {}\n".format(n_envs))
            hyperparamfile.write("RLAgent = {}\n".format(RLAgent))
            hyperparamfile.write("Env = {}\n".format(args.env))
            hyperparamfile.close()
        print("time remaining:", (datetime.now() - begin_perm_time) * (len(hyperparams_list) - test_num))
        test_num += 1
            # env.save("trained_agents/env_" + run_name)
            #
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('spd-say "your program has finished"')
    print("Execution Time:{}".format(datetime.now() - start_time))

    # except StringException as e:
    #     print(e.what())
