def get_small_grid_A2C_ACKTR():
    gamma = categorical('gamma', [0.95, 0.99, 0.999])
    n_steps = categorical('n_steps', [16, 64, 256, 1024])
    lr_schedule = categorical('lr_schedule', ['linear', 'constant'])
    learning_rate = categorical('lr', [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ent_coef = categorical('ent_coef', [1e-7, 1e-5, 1e-3, 1e-1])
    vf_coef = categorical('vf_coef', [0.2, 0.4, 0.6, 0.8])


if __name__ == "__main__":
