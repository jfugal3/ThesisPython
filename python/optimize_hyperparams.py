from hyperparam_opt import hyperparam_optimization
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from helperFun import grav_options

from stable_baselines import PPO2, A2C, ACKTR, DDPG, TRPO, TD3, SAC
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env

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
    '1goal_ee_PD_cont': [myPandaIKWrapper3D, {'has_renderer': False}]
}


def create_env(n_envs, eval_env=False):
    global env_name
    return VecNormalize(make_vec_env(ENVS[env_name][env_id], n_envs=n_envs, env_kwargs=ENVS[env_name][env_kwargs]), norm_obs=False, norm_reward=True)


def create_model(*_args, **kwargs):
    global algo_name, env_name, ALGOS
    return ALGOS[algo_name](policy='MlpPolicy', env=create_env(n_envs=1),
                            verbose=0, **kwargs)

algo_name = 'acktr'
env_name = '1goal_perfect_comp'
# model = create_model()
# print(model())
pd_data_frame = hyperparam_optimization(algo=algo_name, model_fn=create_model, env_fn=create_env, n_jobs=1, n_timesteps=2010)
print(pd_data_frame)
