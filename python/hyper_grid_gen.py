import numpy as np
import os

def get_small_grid_A2C_ACKTR():
    grid = {
        'gamma': [0.95, 0.99, 0.999],
        'n_steps': [16, 64, 256, 1024],
        'lr_schedule': ['linear', 'constant'],
        'lr': [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'ent_coef': [1e-7, 1e-5, 1e-3, 1e-1],
        'vf_coef': [0.2, 0.4, 0.6, 0.8]}

    max_grad_norm = 0.5
    kfac_clip = 0.001
    vf_fisher_coef = 1.0

    hyperparam_list = []
    for gamma in grid['gamma']:
        for n_steps in grid['n_steps']:
            for lr_schedule in grid['lr_schedule']:
                for learning_rate in grid['lr']:
                    for ent_coef in grid['ent_coef']:
                        for vf_coef in grid['vf_coef']:
                            hyperparam_list.append({
                            'gamma': gamma,
                            'n_steps': n_steps,
                            'lr_schedule': lr_schedule,
                            'learning_rate': learning_rate,
                            'ent_coef': ent_coef,
                            'vf_coef': vf_coef,
                            'max_grad_norm': max_grad_norm,
                            'kfac_clip': kfac_clip,
                            'vf_fisher_coef': vf_fisher_coef
                            })
    return hyperparam_list


if __name__ == "__main__":
    hyperparam_list = get_small_grid_A2C_ACKTR()
    print("list size", len(hyperparam_list))
    dir_name = "hyperparamlists_ACKTR"
    os.makedirs(dir_name, exist_ok=True)
    partition = np.linspace(0, len(hyperparam_list), 11, dtype=int)
    for i in range(len(partition) - 1):
        segment = hyperparam_list[partition[i]:partition[i + 1]]
        np.save(os.path.join(dir_name, "list_" + str(i + 1)), segment)
