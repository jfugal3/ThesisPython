from my_panda_free_space_traj import myPandaFreeSpaceTraj
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from generatePatternedTrajectories import print_count
from helperFun import StringException
from helperFun import grav_options
import sys

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise StringException("Usage: training_logs/<path to agent model> <grav-option>")
        # env = myPandaFreeSpaceTraj(has_renderer=True)
        run_name = sys.argv[1]
        grav_option = sys.argv[2]
        if grav_option == "ee_PD_cont":
            env = myPandaIKWrapper3D(has_renderer=True)
        else:
            env = myPandaFreeSpace1Goal(has_renderer=True, grav_option=grav_options[grav_option])
        model = ACKTR.load("training_logs/" + run_name)


        # mean_reward, n_steps = evaluate_policy(model, env, 10)
        # print("avg reward:{}\nnumber of steps:{}".format(mean_reward, n_steps))
        ## Play Agent
        done = False
        obs = env.reset()
        cum_reward = 0
        action_band = 10
        count = 0
        while True:
            if done:
                print("Reward:", cum_reward)
                cum_reward = 0
                obs = env.reset()
                count = 0
            action, _states = model.predict(obs)
            if count % 100:
                print(action)
            obs, reward, done, info = env.step(action)
            count += 1
            cum_reward += reward
            env.render()
    except StringException as e:
        print(e.what())
