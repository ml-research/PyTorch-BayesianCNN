############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
# activation_type = 'softplus'  # 'softplus' or 'relu'
activation_type = 'lrelu'  # 'softplus' or 'relu'
# activation_type = 'rational'  # 'softplus' or 'relu' or 'lrelu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 200
# n_epochs = 5
lr_start = 0.001
num_workers = 4 # set to 0 for debugging with pycharm https://youtrack.jetbrains.com/issue/PY-39489
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
