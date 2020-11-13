# External modules
import numpy as np

# Own modules
from expfam.misc import log_det


#
# Parameter mappings
#


def dim_from_concatenated_vector(v):
    """Returns the value of K for a (K + K*K)-vector."""
    return int(np.sqrt(v.shape[0] + 0.25) - 0.5)


def split_concatenated_vector(v):
    """Split a (K + K*K)-vector into a (K)-vector and a (K*K)-vector."""
    k = dim_from_concatenated_vector(v)
    return v[:k], v[k:]


def concatenated_vector_to_vector_matrix(v):
    """Split a (K + K*K)-vector into a (K)-vector and a (K, K)-matrix."""
    k = dim_from_concatenated_vector(v)
    vector = v[:k]
    matrix = np.reshape(v[k:], (k, k))
    return vector, matrix


def map_from_mu_cov_to_eta(mu, cov):
    """Map parameters from (mu, cov)-space to eta-space."""
    prec = np.linalg.inv(cov)
    return np.concatenate((prec @ mu, -0.5 * np.ravel(prec)))


def map_from_eta_to_mu_cov(eta):
    """Map parameters from eta-space to (mu, cov)-space."""
    eta_0, eta_1 = concatenated_vector_to_vector_matrix(eta)
    cov = np.linalg.inv(-2 * eta_1)
    mu = cov @ eta_0
    return mu, cov


def map_from_mu_prec_to_eta(mu, prec):
    """Map parameters from (mu, prec)-space to eta-space."""
    return np.concatenate((prec @ mu, -0.5 * np.ravel(prec)))


def map_from_eta_to_mu_prec(eta):
    """Map parameters from eta-space to (mu, prec)-space."""
    eta_0, eta_1 = concatenated_vector_to_vector_matrix(eta)
    prec = -2 * eta_1
    mu = np.linalg.solve(prec, eta_0)
    return mu, prec


#
# Exponential family identities
#


def log_h(x):
    k = x.shape[0]
    return -0.5 * k * np.log(2*np.pi)


def t(x):
    return np.concatenate((x, np.ravel(np.outer(x, x))))


def a(eta):
    eta_0, eta_1 = concatenated_vector_to_vector_matrix(v=eta)
    return 0.5 * (eta_0 @ np.linalg.solve(-2*eta_1, eta_0) - log_det(-2*eta_1))


#
# Expected values
#


def ev_t(eta):
    mu, cov = map_from_eta_to_mu_cov(eta)
    ev_t_0 = mu
    ev_t_1 = np.ravel(cov + np.outer(mu, mu))
    return np.concatenate((ev_t_0, ev_t_1))


def split_ev_t(eta):
    mu, cov = map_from_eta_to_mu_cov(eta)
    ev_t_0 = mu
    ev_t_1 = cov + np.outer(mu, mu)
    return ev_t_0, ev_t_1


def ev_x(eta):
    mu, cov = map_from_eta_to_mu_cov(eta)
    return mu


def ev_outer_x(eta):
    mu, cov = map_from_eta_to_mu_cov(eta)
    return cov + np.outer(mu, mu)


def kl_divergence(eta_q, eta_p):
    """KL-divergence{ q(x | eta_q) || p(x | eta_p) }."""
    return ev_t(eta_q) @ (eta_q - eta_p) - a(eta_q) + a(eta_p)


def ev_log_h(eta):
    """E{ log h(x) }."""
    k = dim_from_concatenated_vector(eta)
    return -0.5 * k * np.log(2*np.pi)


def ev_log_p(eta):
    """E{ log p(x | eta) }."""
    return ev_log_h(eta) + ev_t(eta) @ eta - a(eta)


def entropy(eta):
    """-E{ log p(x | eta) }."""
    return -ev_log_p(eta)
