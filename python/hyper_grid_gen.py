import numpy as np
import os

def get_small_grid_ACKTR():
    grid = {
        'gamma': [0.99],
        'n_steps': [32],
        'lr_schedule': ['linear', 'constant', 'double_middle_drop'],
        'lr': [0.3, 0.12, 0.05, 0.02],
        'ent_coef':  [0.0, 0.001, 0.01, 0.1],
        'vf_coef':  [0.25, 0.5, 0.75, 1.0] }

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

def get_small_grid_PPO2():
    grid = {
    'gamma':[0.99],
    'n_steps':[32],
    'ent_coef':[0.0, 0.001, 0.01, 0.1],
    'learning_rate':[0.001, 0.00072, 0.00052, 0.00037, 0.00027, 0.00019, 0.00014, 0.0001],
    'vf_coef':[0.25, 0.5, 0.75, 1.0],
    'max_grad_norm':[0.5],
    'lam':[0.95],
    'nminibatches':[4],
    'noptepochs':[4],
    'cliprange':[0.2],
    'cliprange_vf':[0.2],
    }
    hyperparam_list = []
    for gamma in grid['gamma']:
        for n_steps in grid['n_steps']:
            for ent_coef in grid['ent_coef']:
                for learning_rate in grid['learning_rate']:
                    for vf_coef in grid['vf_coef']:
                        for max_grad_norm in grid['max_grad_norm']:
                            for lam in grid['lam']:
                                for nminibatches in grid['nminibatches']:
                                    for noptepochs in grid['noptepochs']:
                                        for cliprange in grid['cliprange']:
                                            for cliprange_vf in grid['cliprange_vf']:
                                                hyperparam_list.append({
                                                    'gamma': gamma,
                                                    'n_steps': n_steps,
                                                    'ent_coef': ent_coef,
                                                    'learning_rate': learning_rate,
                                                    'vf_coef': vf_coef,
                                                    'max_grad_norm': max_grad_norm,
                                                    'lam': lam,
                                                    'nminibatches': nminibatches,
                                                    'noptepochs': noptepochs,
                                                    'cliprange': cliprange,
                                                    'cliprange_vf': cliprange_vf
                                                })
    return hyperparam_list

if __name__ == "__main__":
    # ACKTR hyperparams
    # hyperparam_list = get_small_grid_ACKTR()
    # dir_name = "hyperparamlists_ACKTR"

    # PPO2 hyperparams
    hyperparam_list = get_small_grid_PPO2()
    dir_name = "hyperparamlists_PPO2"

    print("list size", len(hyperparam_list))
    os.makedirs(dir_name, exist_ok=True)
    partition = np.linspace(0, len(hyperparam_list), 13, dtype=int)
    for i in range(len(partition) - 1):
        segment = hyperparam_list[partition[i]:partition[i + 1]]
        np.save(os.path.join(dir_name, "list_" + str(i + 1)), segment)
