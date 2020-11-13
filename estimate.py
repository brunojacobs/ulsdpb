# Standard library modules
from collections import namedtuple
import argparse
import os

# External modules
import numpy as np

# Own modules
import model.data
import model.fixed
import model.initialization
import model.optimization
import model.prior
import model.state

import settings

# Numpy settings
np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

# Get user arguments
parser = argparse.ArgumentParser()
parser.add_argument('-MODEL', type=str)
parser.add_argument('-M', type=int)
parser.add_argument('-N_ITER', type=int)
parser.add_argument('-N_SAVE_PER', type=int)
parser_args = parser.parse_args()

MODEL = parser_args.MODEL
M = parser_args.M
N_ITER = parser_args.N_ITER
N_SAVE_PER = parser_args.N_SAVE_PER


assert MODEL in ['FULL', 'NO_VAR', 'LDA_X'], \
    'Valid options for MODEL argument are FULL, NO_VAR, or LDA_X'
assert M >= 2, \
    'M should be an integer larger than or equal to 2'

# Process user arguments
EMULATE_LDA_X = None
NO_DYNAMICS = None
NO_REGRESSORS = None

if MODEL == 'FULL':
    EMULATE_LDA_X = False
    NO_DYNAMICS = False
    NO_REGRESSORS = False
    print('Complete ULSDPB-model')
elif MODEL == 'NO_VAR':
    EMULATE_LDA_X = False
    NO_DYNAMICS = True
    NO_REGRESSORS = False
    print('ULSDPB-model without VAR(1) effects')
elif MODEL == 'LDA_X':
    EMULATE_LDA_X = True
    NO_DYNAMICS = False
    NO_REGRESSORS = False
    print('ULSDPB-model restricted to LDA-X')

# Subfolder in the output folder that is M-specific
M_OUTPUT_FOLDER = os.path.join(settings.OUTPUT_FOLDER, 'M' + str(M))

if not os.path.exists(M_OUTPUT_FOLDER):
    os.makedirs(M_OUTPUT_FOLDER)

# Define location for the .csv file with the initial C_JM matrix
INIT_C_JM_FILE = os.path.join(M_OUTPUT_FOLDER, settings.INIT_C_JM_FILENAME)

# Subfolder in the M-specific output folder that is model-specific
MODEL_OUTPUT_FOLDER = os.path.join(M_OUTPUT_FOLDER, MODEL)

if not os.path.exists(MODEL_OUTPUT_FOLDER):
    os.makedirs(MODEL_OUTPUT_FOLDER)

# Create namedtuple with the VI settings
SettingsVI = namedtuple(
    typename='SettingsVI',
    field_names=settings.VI,
)
vi_settings = SettingsVI(**settings.VI) # noqa


# Create namedtuple with the other settings
SettingsMisc = namedtuple(
    typename='SettingsMisc',
    field_names=settings.MISC,
)
misc_settings = SettingsMisc(**settings.MISC) # noqa

# Load the (y_fused_ibn, x, w)-data
y_fused_ibn = np.loadtxt(settings.Y_CSV, dtype=int, delimiter=',')
x = np.loadtxt(settings.X_CSV, dtype=float, delimiter=',')
w = np.asfortranarray(np.loadtxt(settings.W_CSV, dtype=float, delimiter=','))

# Load the C_JM matrix with pseudo-counts from the LDA solution
initial_c_jm = np.loadtxt(INIT_C_JM_FILE, dtype=float, delimiter=',')

# Create a dataset, based on the (y_fused_ibn, x, w)-data
data = model.data.create_dataset(
    emulate_lda_x=EMULATE_LDA_X,
    y_fused_ibn=y_fused_ibn,
    x=x,
    w=w,
)

# Define fixed parameter values, based on the optimization settings
is_fixed, fixed_values = model.fixed.create_fixed(
    emulate_lda_x=EMULATE_LDA_X,
    no_dynamics=NO_DYNAMICS,
    no_regressors=NO_REGRESSORS,
    dim_i=data.dim_i,
    dim_x=data.dim_x,
    dim_w=data.dim_w,
    M=M,
)

# Define the prior parameter values, as specified in model.elbo
prior = model.prior.create_prior(
    is_fixed=is_fixed,
    dim_j=data.dim_j,
    dim_x=data.dim_x,
    dim_w=data.dim_w,
    M=M,
)

# Initialize the variational parameters
initial_state_stub = model.initialization.create_stub_initialization(
    init_ss_mu_q_alpha_ib=settings.INIT_SS_MU_Q_ALPHA_IB,
    init_ss_log_sigma_q_alpha_ib=settings.INIT_SS_LOG_SIGMA_Q_ALPHA_IB,
    c_jm=initial_c_jm,
    prior=prior,
    is_fixed=is_fixed,
    data=data,
    M=M,
)

# Compute the corresponding variational expectations
q = model.state.create_state(
    state_stub=initial_state_stub,
    data=data,
    prior=prior,
    is_fixed=is_fixed,
    fixed_values=fixed_values,
    M=M,
)

np.savez_compressed(
    file=os.path.join(MODEL_OUTPUT_FOLDER, 'data.npz'),
    **data._asdict(),
)

np.savez_compressed(
    file=os.path.join(MODEL_OUTPUT_FOLDER, 'prior.npz'),
    **prior._asdict(),
)

np.savez_compressed(
    file=os.path.join(MODEL_OUTPUT_FOLDER, 'initial_state.npz'),
    **q._asdict(),
)


np.savez_compressed(
    file=os.path.join(MODEL_OUTPUT_FOLDER, 'vi_settings.npz'),
    **vi_settings._asdict(),
)

np.savez_compressed(
    file=os.path.join(MODEL_OUTPUT_FOLDER, 'misc_settings.npz'),
    **misc_settings._asdict(),
)

# Model estimation using variational inference
q, elbo_dict = model.optimization.routine(
    q=q,
    data=data,
    prior=prior,
    is_fixed=is_fixed,
    fixed_values=fixed_values,
    M=M,
    model_output_folder=MODEL_OUTPUT_FOLDER,
    n_iter=N_ITER,
    n_save_per=N_SAVE_PER,
    misc_settings=misc_settings,
    vi_settings=vi_settings,
)
