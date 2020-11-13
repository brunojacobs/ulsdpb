# External modules
import numpy as np

# Own modules
from expfam.misc import log_det, log_mvar_gamma, mvar_digamma


#
# Parameter mappings
#


def dim_from_concatenated_vector(v):
    """Returns the value of K for a (1 + K*K)-vector."""
    return int(np.sqrt(v.shape[0] - 1))


def concatenated_vector_to_scalar_matrix(v):
    """Split a (1 + K*K)-vector into a scalar and a (K, K)-matrix."""
    k = dim_from_concatenated_vector(v)
    scalar = v[0]
    matrix = np.reshape(v[1:], (k, k))
    return scalar, matrix


def map_from_n_v_to_eta(n, v):
    """Map parameters from (n, v)-space to eta-space."""
    return np.hstack((0.5 * n, -0.5 * np.ravel(np.linalg.inv(v))))


def map_from_eta_to_n_v(eta):
    """Map parameters from eta-space to (n, v)-space."""
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    n = 2 * eta_0
    v = np.linalg.inv(-2 * eta_1)
    return n, v


#
# Exponential family identities
#


def log_h(x):
    k = x.shape[0]
    return -0.5 * (k + 1) * log_det(x)


def t(x):
    return np.hstack((log_det(x), np.ravel(x)))


def a(eta):
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    k = eta_1.shape[0]
    return log_mvar_gamma(k, eta_0) - eta_0 * log_det(-eta_1)


#
# Expected values
#


def ev_t(eta):
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    k = eta_1.shape[0]
    ev_t_0 = mvar_digamma(dim=k, arg=eta_0) - log_det(-eta_1)
    ev_t_1 = eta_0 * np.ravel(np.linalg.inv(-eta_1))
    return np.hstack((ev_t_0, ev_t_1))


def split_ev_t(eta):
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    k = eta_1.shape[0]
    ev_t_0 = mvar_digamma(dim=k, arg=eta_0) - log_det(-eta_1)
    ev_t_1 = eta_0 * np.linalg.inv(-eta_1)
    return ev_t_0, ev_t_1


def ev_log_det_x(eta):
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    k = eta_1.shape[0]
    return mvar_digamma(dim=k, arg=eta_0) - log_det(-eta_1)


def ev_x(eta):
    eta_0, eta_1 = concatenated_vector_to_scalar_matrix(eta)
    ev_t_1 = eta_0 * np.linalg.inv(-eta_1)
    return ev_t_1


def ev_inv_x(eta):
    n, v = map_from_eta_to_n_v(eta=eta)
    mean_inv_x = np.linalg.inv(v) / (n - v.shape[0] - 1)
    return mean_inv_x


def kl_divergence(eta_q, eta_p):
    """KL-divergence{ q(x | eta_q) || p(x | eta_p) }."""
    return ev_t(eta_q) @ (eta_q - eta_p) - a(eta_q) + a(eta_p)
