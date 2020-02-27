# from myPandaDoorEnv import myPandaDoor
from my_panda_free_space_traj import myPandaFreeSpaceTraj
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
import my_panda_free_space_traj
from stable_baselines import PPO2, A2C, ACKTR, DDPG, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecNormalize
import stable_baselines.common.cmd_util
from stable_baselines import logger
from helperFun import StringException
from helperFun import grav_options

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
                _locals['self'].save(log_dir + 'best_model.pkl')
                env.save(log_dir + "vec_normalize.pkl")
            _locals['self'].save(checkpoint_dir + "model_after_" + str(x[-1]) + ".pkl")
            env.save(checkpoint_dir + "vec_normalize_env_after_" + str(x[-1]) + ".pkl")

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
    try:
        if len(sys.argv) < 3:
            raise StringException("Usage: <run-name> <gravity-option> <render-monitor (optional)>")

        run_name = sys.argv[1]
        grav_option = sys.argv[2]

        if grav_option not in grav_options:
            raise StringException(grav_option + "is not a valid gravity option." + "\nValid options:\n" + str(grav_options))

        rendering = False
        if len(sys.argv) > 3:
            render_mode = sys.argv[3]
            if render_mode == "True":
                rendering = True

        start_time = datetime.now()

        log_dir = os.path.join("training_logs", run_name)
        checkpoint_dir = os.path.join(log_dir, "model_checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.configure(log_dir)
        if grav_option == "ee_PD_cont":
            env_handle = myPandaIKWrapper3D
            env_kwargs = {"has_renderer" : False}
            # print("here1")
        else:
            env_handle = myPandaFreeSpace1Goal
            env_kwargs = {"grav_option" : grav_options[grav_option], "has_renderer" : False}
            # print("here2")

        # env = env_handle(*env_kwargs)
        # print(env.action_space)
        # print(env)

        env = stable_baselines.common.make_vec_env(env_handle, n_envs=7, monitor_dir=log_dir, env_kwargs=env_kwargs)
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        RLAgent = PPO2
        model = RLAgent('MlpPolicy', env, verbose=1).learn(total_timesteps=int(5e5), callback=callback)
        model.save("trained_agents/" + run_name)
        env.save("trained_agents/env_" + run_name)
        #
        duration = 1  # seconds
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        os.system('spd-say "your program has finished"')
        print("Execution Time:{}".format(datetime.now() - start_time))

    except StringException as e:
        print(e.what())
