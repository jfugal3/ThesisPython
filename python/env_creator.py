from my_panda_IK_wrapper_3d import myPandaIKWrapper3D
from my_panda_free_space_1goal import myPandaFreeSpace1Goal
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common import make_vec_env
from helperFun import grav_options

env_id = 0
env_kwargs = 1

ENVS = {
    '1goal_no_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["no_comp"]}],
    '1goal_perfect_comp': [myPandaFreeSpace1Goal, {'has_renderer': False, 'grav_option': grav_options["perfect_comp"]}],
    '1goal_ee_PD_cont': [myPandaIKWrapper3D, {'has_renderer': False}],
    # 'panda_lift_no_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['no_comp']}],
    # 'panda_lift_perfect_comp': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['perfect_comp']}],
    # 'panda_lift_ee_PD_cont': [myPandaLift, {'has_renderer': False, 'grav_option': grav_options['ee_PD_cont']}]
}


def create_env(n_envs, env_name=None, log_dir=None):
    return VecNormalize(make_vec_env(ENVS[env_name][env_id], n_envs=n_envs, env_kwargs=ENVS[env_name][env_kwargs], monitor_dir=log_dir), norm_obs=False, norm_reward=True)
