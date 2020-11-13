# External modules
import numpy as np
import numba

# Own modules
from expfam import settings


#
# Parameter mappings
#


@numba.jit(**settings.NUMBA_OPTIONS)
def dim_from_concatenated_vector(v):
    """Returns the value of K for a (2K)-vector."""
    return np.int(v.shape[0] / 2)


@numba.jit(**settings.NUMBA_OPTIONS)
def split_concatenated_vector(v):
    """Split a (2K,)-vector into two (K,)-vectors."""
    k = dim_from_concatenated_vector(v)
    return v[:k], v[k:]


def map_from_mu_sigma_sq_to_eta(mu, sigma_sq):
    """Map parameters from (mu, sigma_sq)-space to eta-space."""
    tau = sigma_sq**-1
    return np.hstack((tau * mu, -0.5 * tau))


@numba.jit(**settings.NUMBA_OPTIONS)
def map_from_eta_to_mu_sigma_sq(eta):
    """Map parameters from eta-space to (mu, sigma_sq)-space."""
    eta_0, eta_1 = split_concatenated_vector(eta)
    sigma_sq = (-2 * eta_1)**-1
    mu = sigma_sq * eta_0
    return mu, sigma_sq


def map_from_mu_tau_to_eta(mu, tau):
    """Map parameters from (mu, tau)-space to eta-space."""
    return np.hstack((tau * mu, -0.5 * tau))


def map_from_eta_to_mu_tau(eta):
    """Map parameters from eta-space to (mu, tau)-space."""
    eta_0, eta_1 = split_concatenated_vector(eta)
    tau = -2 * eta_1
    mu = eta_0 / tau
    return mu, tau


#
# Exponential family identities
#


def log_h(x):
    k = x.shape[0]
    return -0.5 * k * np.log(2*np.pi)


def t(x):
    return np.hstack((x, x*x))


@numba.jit(**settings.NUMBA_OPTIONS)
def a(eta):
    eta_0, eta_1 = split_concatenated_vector(v=eta)
    return 0.5 * np.sum(eta_0**2 / (-2 * eta_1) - np.log(-2 * eta_1))


#
# Expected values
#


@numba.jit(**settings.NUMBA_OPTIONS)
def ev_t(eta):
    mu, sigma_sq = map_from_eta_to_mu_sigma_sq(eta)
    return np.hstack((mu, sigma_sq + mu*mu))


def ev_x(eta):
    mu, sigma_sq = map_from_eta_to_mu_sigma_sq(eta)
    return mu


def ev_x_sq(eta):
    mu, sigma_sq = map_from_eta_to_mu_sigma_sq(eta)
    return sigma_sq + mu*mu


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
