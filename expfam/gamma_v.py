# External modules
import numpy as np
from scipy.special import gammaln, digamma


#
# Parameter mappings
#

def dim_from_concatenated_vector(v):
    """Returns the value of K for a (2K)-vector."""
    return np.int(v.shape[0] / 2)


def split_concatenated_vector(v):
    """Split a (2K,)-vector into two (K,)-vectors."""
    k = dim_from_concatenated_vector(v)
    return v[:k], v[k:]


def map_from_alpha_beta_to_eta(alpha, beta):
    """Map parameters from (alpha, beta)-space to eta-space."""
    return np.concatenate((alpha, -beta))


def map_from_eta_to_alpha_beta(eta):
    """Map parameters from eta-space to (alpha, beta)-space."""
    eta_0, eta_1 = split_concatenated_vector(eta)
    alpha = eta_0
    beta = -eta_1
    return alpha, beta


#
# Exponential family identities
#


def log_h(x):
    return -np.sum(np.log(x))


def t(x):
    return np.concatenate((np.log(x), x))


def a(eta):
    eta_0, eta_1 = split_concatenated_vector(eta)
    return np.sum(gammaln(eta_0) - eta_0 * np.log(-eta_1))


#
# Expected values
#


def ev_t(eta):
    eta_0, eta_1 = split_concatenated_vector(eta)
    ev_t_0 = digamma(eta_0) - np.log(-eta_1)
    ev_t_1 = eta_0 / -eta_1
    return np.concatenate((ev_t_0, ev_t_1))


def split_ev_t(v):
    return split_concatenated_vector(v)


def ev_log_x(eta):
    eta_0, eta_1 = split_concatenated_vector(eta)
    ev_t_0 = digamma(eta_0) - np.log(-eta_1)
    return ev_t_0


def ev_x(eta):
    eta_0, eta_1 = split_concatenated_vector(eta)
    ev_t_1 = eta_0 / -eta_1
    return ev_t_1


def ev_inv_x(eta):
    alpha, beta = map_from_eta_to_alpha_beta(eta)
    assert np.all(alpha > 1)
    return beta / (alpha - 1)


def kl_divergence(eta_q, eta_p):
    """KL-divergence{ q(x | eta_q) || p(x | eta_p) }."""
    return ev_t(eta_q) @ (eta_q - eta_p) - a(eta_q) + a(eta_p)
