# External modules
import numpy as np
from scipy.special import gammaln, digamma
import numba

# Own modules
from expfam import settings


#
# Miscellaneous vector functions
#


@numba.jit(**settings.NUMBA_OPTIONS)
def log_sum_exp(v):
    """Returns log(sum(exp(v))) for a vector v."""
    max_v_value = v[0]
    for k in range(1, v.size):
        if v[k] > max_v_value:
            max_v_value = v[k]
    return max_v_value + np.log(np.sum(np.exp(v - max_v_value)))


def softmax(v):
    """Returns softmax(v) for a vector v."""
    return np.exp(v) / np.sum(np.exp(v))


#
# Miscellaneous matrix functions
#


def log_det(m):
    """The (natural) log of the determinant of a matrix."""
    sign_determinant, log_determinant = np.linalg.slogdet(m)
    return log_determinant


def is_symmetric(m):
    """Returns true if the matrix is symmetric."""
    return np.allclose(m, m.T)


def corr_from_cov(cov):
    """Compute correlation matrix from a covariance matrix."""
    sigma_sq = np.diag(cov)
    return np.diag(sigma_sq**-0.5) @ cov @ np.diag(sigma_sq**-0.5)


def corr_from_prec(prec):
    """Compute correlation matrix from a precision matrix."""
    return corr_from_cov(np.linalg.inv(prec))


def inv_corr_from_cov(cov):
    """Compute the inverse correlation matrix from a covariance matrix."""
    return inv_corr_from_prec(np.linalg.inv(cov))


def inv_corr_from_prec(prec):
    """Compute the inverse correlation matrix from a precision matrix."""
    tau = np.diag(prec)
    return np.diag(tau**-0.5) @ prec @ np.diag(tau**-0.5)


#
# Multivariate beta and multivariate gamma
#


def log_mvar_beta(arg):
    """The (natural) log multivariate beta function."""
    return np.sum(gammaln(arg)) - gammaln(np.sum(arg))


def log_mvar_gamma(dim, arg):
    """The (natural) log multivariate gamma function."""
    result = dim * (dim - 1) / 4 * np.log(np.pi)
    for j in range(dim):
        result += gammaln(arg - j / 2)
    return result


def mvar_digamma(dim, arg):
    """The multivariate digamma function."""
    result = 0
    for i in range(1, dim + 1):
        result += digamma(arg + (1 - i) / 2)
    return result
