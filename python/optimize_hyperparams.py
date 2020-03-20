import os
import argparse
from my_panda_lift import myPandaLift
from hyperparam_opt import hyperparam_optimization
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from helperFun import grav_options

from stable_baselines import PPO2, A2C, ACKTR, DDPG, TRPO, TD3, SAC
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env
from datetime import datetime
from functools import partial

ALGOS = {
    'a2c': A2C,
    'acktr': ACKTR,
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}

env_id = 0
env_kwargs = 1

ENVS = {
    '1goal_no_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["no_comp"]}],
    '1goal_perfect_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["perfect_comp"]}],
    '1goal_ee_PD_cont': [myPandaIKWrapper3D, {'has_renderer': False}],
    'panda_lift_no_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['no_comp']}],
    'panda_lift_perfect_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['perfect_comp']}],
    'panda_lift_ee_PD_cont': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['ee_PD_cont']}]
}

def create_env(n_envs, eval_env=False):
    global env_name
    return VecNormalize(make_vec_env(ENVS[env_name][env_id], n_envs=n_envs, env_kwargs=ENVS[env_name][env_kwargs], monitor_dir=log_dir), norm_obs=False, norm_reward=True)

# def create_env(n_envs, eval_env=False, env_name=None, log_dir=None):
#     return VecNormalize(make_vec_env(ENVS[env_name][env_id], n_envs=n_envs, env_kwargs=ENVS[env_name][env_kwargs], monitor_dir=log_dir), norm_obs=False, norm_reward=True)

# def env_fn(n_envs, eval_env):
#     global env_name
#     return create_env(n_envs, eval_env, env_name)


def create_model(*_args, **kwargs):
    global algo_name, env_name, ALGOS
    return ALGOS[algo_name](policy='MlpPolicy', env=create_env(n_envs=1),
                            verbose=0, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize RL hyperparameters for specific environments")
    parser.add_argument("-a", "--RLAgent", help="RL algorithm", required=True, choices=ALGOS.keys())
    parser.add_argument("-e", "--env", help="Environment", required=True, choices=ENVS.keys())
    parser.add_argument("-j", "--n_jobs", type=int, help="number of jobs to run simultainiously", required=False, default=5)
    parser.add_argument("-tri", "--n_trials", type=int, help="number of hyperparamiterizations to try", required=False, default=1000)
    parser.add_argument("-ts", "--timesteps", type=int, help="number of timesteps to run each training session", required=False, default=int(1e5))
    args = parser.parse_args()



    algo_name = args.RLAgent
    env_name = args.env
    n_jobs = args.n_jobs
    n_trials = args.n_trials
    n_timesteps = args.timesteps


    dir_name = "hyperparams"
    file_name = os.path.join(dir_name, algo_name + "_" + env_name + ".csv")
    # model = create_model()
    # print(model())
    start_time = datetime.now()
    # env_fn = lambda n_envs, eval_env, env_name=env_name: create_env(n_envs=n_envs, eval_env=eval_env, env_name=env_name)
    env_fn = partial(create_env, env_name=env_name)
    pd_data_frame = hyperparam_optimization(algo=algo_name, model_fn=create_model, env_fn=env_fn, n_jobs=n_jobs, n_timesteps=n_timesteps, n_trials=n_trials)
    print(pd_data_frame)
    print("Saving data to$", file_name)
    pd_data_frame.to_csv(file_name)
    print("Elasped time", datetime.now() - start_time)
