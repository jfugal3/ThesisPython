# from myPandaDoorEnv import myPandaDoor
from my_panda_lift import myPandaLift
# from my_panda_free_space_traj import myPandaFreeSpaceTraj
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
import my_panda_free_space_traj
from stable_baselines import A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecNormalize
import stable_baselines.common.cmd_util
from stable_baselines import logger
from helperFun import StringException
from helperFun import grav_options
from env_creator import create_env
from env_creator import ENVS
import argparse

# from stable_baselines.common import make_vec_env
import gym
import os
import numpy as np
from datetime import datetime
import pandas as pd
import sys

#
# def my_load_results(log_dir):
#     dat = pd.read_csv(os.join(log_dir, "progress.csv"), dtype={"total_timesteps": int, "ep_reward_mean": float}, usecols=["total_timesteps", "ep_reward_mean"])
#     x, y = dat["total_timesteps"].to_numpy(), np.nan_to_num(dat["ep_reward_mean"].to_numpy())



best_mean_reward, n_steps = -np.inf, 0
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, env, log_dir, checkpoint_dir
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + '/best_model.pkl')
                env.save(log_dir + "vec_normalize.pkl")
            _locals['self'].save(checkpoint_dir + "/model_after_" + str(x[-1]) + ".pkl")
            # env.save(checkpoint_dir + "/vec_normalize_env_after_" + str(x[-1]) + ".pkl")

            global rendering
            if rendering:
                global RLAgent, env_handle, env_kwargs
                model = RLAgent.load(log_dir + "model_after_" + str(x[-1]) + ".pkl")
                env = env_handle(*env_kwargs)
                print(env.action_space)
                for i in range(3):
                    done = False
                    obs = env.reset()
                    cum_reward = 0
                    while not done:
                        action, _states = model.predict(obs)
                        obs, reward, done, info = env.step(action)
                        cum_reward += reward
                        env.render()
                    print(action)
                    print("Reward:", cum_reward)
                del env
    # print(n_steps)
    n_steps += 1
    # Returning False will stop training early
    return True

if __name__ == "__main__":
    # try:
    #     if len(sys.argv) < 3:
    #         raise StringException("Usage: <run-name> <gravity-option> <render-monitor (optional)>")
    #
    #     run_name = sys.argv[1]
    #     grav_option = sys.argv[2]
    #
    #     if grav_option not in grav_options:
    #         raise StringException(grav_option + "is not a valid gravity option." + "\nValid options:\n" + str(grav_options))
    kwargs = {'gamma':0.999, 'n_steps':20, 'ent_coef':0.01, 'vf_coef':0.25, 'vf_fisher_coef':1.0, 'learning_rate':1.00000000e-05, 'max_grad_norm':0.5, 'kfac_clip':0.001, 'lr_schedule':'linear', 'kfac_update':1}
    parser = argparse.ArgumentParser("Train an RL Agent")
    parser.add_argument("-rn", "--run_name", required=True)
    parser.add_argument("-env", required=True, choices=ENVS.keys())
    parser.add_argument("-n_envs", type=int, required=False, default=1)
    parser.add_argument("-ts", "--timesteps", type=int, help="number of timesteps to run each training session", required=False, default=int(5e5))
    parser.add_argument("--rendering", type=bool, required=False, default=False)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-mgn", "--max_grad_norm", type=float, required=False, default=kwargs["max_grad_norm"])
    parser.add_argument("-kfac_clip", type=float, required=False, default=kwargs["kfac_clip"])
    parser.add_argument("-vf_coef", type=float, required=False, default=kwargs["vf_coef"])
    parser.add_argument("-ent_coef", type=float, required=False, default=kwargs["ent_coef"])
    args = parser.parse_args()
    rendering = args.rendering
    total_timesteps = args.timesteps
    run_name = args.run_name
    lr = args.lr
    kwargs['learning_rate'] = lr
    kwargs['max_grad_norm'] = args.max_grad_norm
    kwargs['kfac_clip'] = args.kfac_clip
    kwargs['vf_coef'] = args.vf_coef
    kwargs['ent_coef'] = args.ent_coef

    start_time = datetime.now()

    log_dir = os.path.join("training_logs", run_name)
    checkpoint_dir = os.path.join(log_dir, "model_checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.configure(log_dir)

    env = create_env(n_envs=args.n_envs, env_name=args.env, log_dir=log_dir)
    RLAgent = ACKTR
    hyperparamfilename = os.path.join(log_dir, "hyperparams.txt")
    hyperparamfile = open(hyperparamfilename, 'w')
    hyperparamfile.write(str(kwargs))
    hyperparamfile.write("\nn_envs = {}\n".format(args.n_envs))
    hyperparamfile.write("RLAgent = {}\n".format(RLAgent))
    hyperparamfile.close()
    model = RLAgent('MlpPolicy', env, verbose=1, **kwargs).learn(total_timesteps=total_timesteps, callback=callback)
    model.save(log_dir + "final_agent.pkl")
    # env.save("trained_agents/env_" + run_name)
    print(kwargs)
    #
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('spd-say "your program has finished"')
    print("Execution Time:{}".format(datetime.now() - start_time))

    # except StringException as e:
    #     print(e.what())
