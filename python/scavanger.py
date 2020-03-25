import warnings
warnings.simplefilter("ignore")
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
import my_panda_free_space_traj
from stable_baselines import A2C, ACKTR
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
    # parser = argparse.ArgumentParser("Grid Search")
    # parser.add_argument("-ld", "--log_dir", required=True)
    # parser.add_argument("-hd", "--hyperparamfile", required=True)
    # parser.add_argument("-env", required=True, choices=['1goal_no_comp', '1goal_perfect_comp'])
    # parser.add_argument("-n_envs", type=int, required=False, default=1)
    # parser.add_argument("-ts", type=int, required=True)
    # args = parser.parse_args()

    # top_log_dir = args.log_dir
    # hyperparams_list = np.load(args.hyperparamfile, allow_pickle=True)
    # env_name = args.env
    # n_envs = args.n_envs
    # timesteps = args.ts

    ##################################################################################################################################################################################################
    top_log_dir = "perfect_comp_100_training_sessions"
    hyperparams = {'ent_coef': 0.001, 'vf_fisher_coef': 1.0, 'n_steps': 32, 'kfac_clip': 0.001, 'max_grad_norm': 0.5, 'lr_schedule': 'constant', 'gamma': 0.99, 'learning_rate': 0.05, 'vf_coef': 0.75}
    env_name = "1goal_perfect_comp"
    ##################################################################################################################################################################################################
    n_envs = 4
    timesteps = 250000
    RLAgent = ACKTR

    start_time = datetime.now()

    top_log_dir = os.path.join("training_logs", top_log_dir)
    os.makedirs(top_log_dir, exist_ok=True)
    run_num = 1
    while not os.path.exists(os.path.join(top_log_dir, "run_100_monitor_dir")):
        print("Beginning run", run_num, "of 100")
        # begin_perm_time = datetime.now()
        run_dir = os.path.join(top_log_dir, "run_" + str(run_num) + "_monitor_dir")
        run_num += 1
        hyperparamfilename = os.path.join(run_dir, "hyperparams.txt")
        if os.path.exists(run_dir):
            continue

        os.makedirs(run_dir, exist_ok=True)
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
        hyperparamfile.write("\nenv = {}\n".format(env_name))
        hyperparamfile.write("RLAgent = {}\n".format(RLAgent))
        hyperparamfile.close()
        # print("time remaining:", (datetime.now() - begin_perm_time) * (len(hyperparams_list) - test_num))
            # env.save("trained_agents/env_" + run_name)
            #
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('spd-say "your program has finished"')
    print("Execution Time:{}".format(datetime.now() - start_time))

    # except StringException as e:
    #     print(e.what())
