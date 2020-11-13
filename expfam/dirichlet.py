# External modules
import numpy as np
from scipy.special import digamma

# Own modules
from expfam.misc import log_mvar_beta


#
# Parameter mappings
#


def map_from_eta_to_alpha(eta):
    """Map parameters from eta-space to alpha-space."""
    return eta


def map_from_alpha_to_eta(alpha):
    """Map parameters from alpha-space to eta-space."""
    return alpha


#
# Exponential family identities
#


def log_h(x):
    return -np.sum(np.log(x))


def t(x):
    return np.log(x)


def a(eta):
    return log_mvar_beta(eta)


#
# Expected values
#


def ev_t(eta):
    return digamma(eta) - digamma(np.sum(eta))


def ev_x(eta):
    return eta / np.sum(eta)


def kl_divergence(eta_q, eta_p):
    """KL-divergence{ q(x | eta_q) || p(x | eta_p) }."""
    return ev_t(eta_q) @ (eta_q - eta_p) - a(eta_q) + a(eta_p)
