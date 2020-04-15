from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env
from helperFun import grav_options
import numpy as np

env_id = 0
env_kwargs = 1

ENVS = {
    '1goal_no_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["no_comp"]}],

    '1goal_perfect_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["perfect_comp"]}],

    '1goal_ee_PD_cont': [myPandaIKWrapper3D, {'has_renderer': False}],

    '1goal_no_comp_2': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["no_comp"], 'randomize_initialization_std_dev': 0.05,
                                                'init_qpos': np.array([-np.pi/4, np.pi / 16.0, 0.00, -np.pi / 2.0 -np.pi/6, 0.00, np.pi - 0.2, np.pi / 4]),
                                                'target_xyz': [0.4, 0.4, 1.0]}],

    '1goal_perfect_comp_2': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["perfect_comp"], 'randomize_initialization_std_dev': 0.05,
                                                'init_qpos': np.array([-np.pi/4, np.pi / 16.0, 0.00, -np.pi / 2.0 -np.pi/6, 0.00, np.pi - 0.2, np.pi / 4]),
                                                'target_xyz': [0.4, 0.4, 1.0]}],

    '1goal_no_comp_3': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["no_comp"], 'randomize_initialization_std_dev': 0.05,
                                                'init_qpos': np.array([-np.pi/4, np.pi / 16.0, 0.00, -np.pi / 2.0 -np.pi/6, 0.00, np.pi - 0.2, np.pi / 4]),
                                                'target_xyz': [0.3, 0.3, 1.5]}],

    '1goal_perfect_comp_3': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["perfect_comp"], 'randomize_initialization_std_dev': 0.05,
                                                'init_qpos': np.array([-np.pi/4, np.pi / 16.0, 0.00, -np.pi / 2.0 -np.pi/6, 0.00, np.pi - 0.2, np.pi / 4]),
                                                'target_xyz': [0.3, 0.3, 1.5]}],
    # 'panda_lift_no_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['no_comp']}],
    # 'panda_lift_perfect_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['perfect_comp']}],
    # 'panda_lift_ee_PD_cont': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['ee_PD_cont']}]
}


def create_env(n_envs, env_name=None, log_dir=None):
    return VecNormalize(make_vec_env(ENVS[env_name][env_id], n_envs=n_envs, env_kwargs=ENVS[env_name][env_kwargs], monitor_dir=log_dir), norm_obs=False, norm_reward=True)

def create_one_env(env_name, has_renderer=False):
    args = ENVS[env_name][env_kwargs].copy()
    args['has_renderer'] = has_renderer
    print(args)
    return ENVS[env_name][env_id](**args)
