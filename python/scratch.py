from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ,ent_coef,gamma,kfac_clip,learning_rate,lr_schedule,max_grad_norm,n_steps,vf_coef,vf_fisher_coef,
# 1,0.001,0.99,0.001,0.12,constant,0.5,32,1.0,1.0,25324.03,16414.65,
# dir_name = "training_logs/no_grav_comp_grid_search/ent_coef-0.001__gamma-0.99__kfac_clip-0.001__learning_rate-0.12__lr_schedule-constant__max_grad_norm-0.5__n_steps-32__vf_coef-1.0__vf_fisher_coef-1.0"
# dir_name = "training_logs/no_comp_100_training_sessions"

# ,ent_coef,gamma,kfac_clip,learning_rate,lr_schedule,max_grad_norm,n_steps,vf_coef,vf_fisher_coef,
# 1,0.001,0.99,0.001,0.05,constant,0.5,32,0.75,1.0,30680.19,20330.87,
# dir_name = "training_logs/perfect_grav_comp_grid_search/ent_coef-0.001__gamma-0.99__kfac_clip-0.001__learning_rate-0.05__lr_schedule-constant__max_grad_norm-0.5__n_steps-32__vf_coef-0.75__vf_fisher_coef-1.0"
dir_name = "training_logs/perfect_comp_100_training_sessions"
y_arr = []
for dir in os.listdir(dir_name):
    if dir.endswith("monitor_dir"):
        x, y = ts2xy(load_results(os.path.join(dir_name,dir)), 'timesteps')
        y_arr.append(y)
y_block = np.vstack(y_arr)
print("mean:", np.around(np.mean(y_block),2))
print("std:", np.around(np.std(y_block),2))
