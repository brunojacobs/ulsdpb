import os

# I/O FOLDERS
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "output"

# DATA FILES
Y_CSV = os.path.join(INPUT_FOLDER, "y.csv")
X_CSV = os.path.join(INPUT_FOLDER, "x.csv")
W_CSV = os.path.join(INPUT_FOLDER, "w.csv")

# NOTE: INIT_C_JM_FILENAME SHOULD BE PLACED IN OUTPUT_FOLDER/$M
INIT_C_JM_FILENAME = 'initial_c_jm.csv'

# The initial step size for the gradient of mu_q_alpha_ib
INIT_SS_MU_Q_ALPHA_IB = 0.125

# The initial step size for the gradient of log_sigma_q_alpha_ib
INIT_SS_LOG_SIGMA_Q_ALPHA_IB = 0.125

# VI OPTIMIZATION SETTINGS
VI = {
    # Number of subiterations per customer (denoted by L in the paper)
    'n_q_i_steps': 25,
    # Settings for adaptive step sizes
    'ss_factor': 1.125,
    'ss_min': 1e-6,
    'ss_max': 1.0,
    'min_elbo_diff': 1e-6,
}

# MISCELLANEOUS SETTINGS
MISC = {
    'n_print_per': 1,
    'check_state_consistency': False,  # if True slows down code significantly
    'profile_code': False,
}
