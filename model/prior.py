#
# Description:
#   Contains the Prior structure for the ULSDPB model.
#
# Functions:
#   create_prior: creates the prior structure for the ULSDPB model based on
#   the dimensions of the model.


# Standard library modules
from collections import namedtuple

# External modules
import numpy as np

# Own modules
from expfam import dirichlet
from expfam import gamma_v
from expfam import mvn
from expfam import normal_v
from expfam import wishart


Prior = namedtuple(
    'Prior',
    [
        # phi
        'phi_alpha',
        'phi_eta',
        'phi_a',
        # tau_alpha
        'tau_alpha_alpha',
        'tau_alpha_beta',
        'tau_alpha_eta',
        'tau_alpha_a',
        # mu_kappa
        'mu_kappa_mu',
        'mu_kappa_sigma',
        'mu_kappa_eta',
        'mu_kappa_a',
        # lambda_kappa
        'lambda_kappa_n',
        'lambda_kappa_v',
        'lambda_kappa_eta',
        'lambda_kappa_a',
        # beta
        'beta_mu',
        'beta_sigma',
        'beta_lambda',
        'beta_eta',
        'beta_a',
        # gamma
        'gamma_mu',
        'gamma_sigma',
        'gamma_lambda',
        'gamma_eta',
        'gamma_a',
        # rho
        'rho_mu',
        'rho_sigma',
        'rho_lambda',
        'rho_eta',
        'rho_a',
        # delta
        'delta_mu',
        'delta_sigma_sq',
        'delta_eta',
        'delta_a',
        # delta_kappa
        'delta_kappa_mu',
        'delta_kappa_sigma_sq',
        'delta_kappa_eta',
        'delta_kappa_a',
        # delta_betta
        'delta_beta_mu',
        'delta_beta_sigma_sq',
        'delta_beta_eta',
        'delta_beta_a',
        # delta_gamma
        'delta_gamma_mu',
        'delta_gamma_sigma_sq',
        'delta_gamma_eta',
        'delta_gamma_a',
    ],
    defaults=[
        # phi
        None,  # 'phi_alpha',
        None,  # 'phi_eta',
        None,  # 'phi_a',
        # tau_alpha
        None,  # 'tau_alpha_alpha',
        None,  # 'tau_alpha_beta',
        None,  # 'tau_alpha_eta',
        None,  # 'tau_alpha_a',
        # mu_kappa
        None,  # 'mu_kappa_mu',
        None,  # 'mu_kappa_sigma',
        None,  # 'mu_kappa_eta',
        None,  # 'mu_kappa_a',
        # lambda_kappa
        None,  # 'lambda_kappa_n',
        None,  # 'lambda_kappa_v',
        None,  # 'lambda_kappa_eta',
        None,  # 'lambda_kappa_a',
        # beta
        None,  # 'beta_mu',
        None,  # 'beta_sigma',
        None,  # 'beta_lambda',
        None,  # 'beta_eta',
        None,  # 'beta_a',
        # gamma
        None,  # 'gamma_mu',
        None,  # 'gamma_sigma',
        None,  # 'gamma_lambda',
        None,  # 'gamma_eta',
        None,  # 'gamma_a',
        # rho
        None,  # 'rho_mu',
        None,  # 'rho_sigma',
        None,  # 'rho_lambda',
        None,  # 'rho_eta',
        None,  # 'rho_a',
        # delta
        None,  # 'delta_mu',
        None,  # 'delta_sigma_sq',
        None,  # 'delta_eta',
        None,  # 'delta_a',
        # delta_kappa
        None,  # 'delta_kappa_mu',
        None,  # 'delta_kappa_sigma_sq',
        None,  # 'delta_kappa_eta',
        None,  # 'delta_kappa_a',
        # delta_beta
        None,  # 'delta_beta_mu',
        None,  # 'delta_beta_sigma_sq',
        None,  # 'delta_beta_eta',
        None,  # 'delta_beta_a',
        # delta_gamma
        None,  # 'delta_gamma_mu',
        None,  # 'delta_gamma_sigma_sq',
        None,  # 'delta_gamma_eta',
        None,  # 'delta_gamma_a',
    ]
)


def create_prior(
        is_fixed,
        dim_j,
        dim_x,
        dim_w,
        M,
):

    #
    # Parameterization
    #

    # p(phi) ~ Dirichlet_{dim_j}(alpha)
    # Note: This is a Dirichlet __matrix__
    phi_dist = dirichlet
    phi_alpha = np.ones((dim_j, M)) / dim_j

    # p(tau_alpha) ~ Gamma_M(alpha, beta)
    # Note: This is a Gamma __vector__
    tau_alpha_dist = gamma_v
    tau_alpha_alpha = np.ones(M)
    tau_alpha_beta = np.ones(M)

    # p(mu_kappa) ~ MVN_M(mu, Sigma)
    mu_kappa_dist = mvn
    mu_kappa_mu = np.zeros(M)
    mu_kappa_sigma = np.identity(M)

    # p(lambda_kappa) ~ Wishart_M(n, V)
    lambda_kappa_dist = wishart
    lambda_kappa_n = 2 * M
    lambda_kappa_v = np.identity(M) / (2 * M)

    # p(beta_m) ~ MVN_{dim_x}(mu, Sigma)
    beta_dist = mvn
    beta_mu = np.zeros(dim_x)
    beta_sigma = np.identity(dim_x)

    # p(gamma_m) ~ MVN_{dim_w}(mu, Sigma)
    gamma_dist = mvn
    gamma_mu = np.zeros(dim_w)
    gamma_sigma = np.identity(dim_w)

    # p(rho_m) ~ MVN_M(mu, Sigma)
    rho_dist = mvn
    rho_mu = np.zeros(M)
    rho_sigma = np.identity(M)

    # p(delta) ~ Normal_M(mu, sigma_sq)
    delta_dist = normal_v
    delta_mu = np.zeros(M)
    delta_sigma_sq = np.ones(M)

    # p(delta_kappa) ~ Normal_1(mu, sigma_sq)
    delta_kappa_dist = normal_v
    delta_kappa_mu = np.array([1.0])
    delta_kappa_sigma_sq = np.array([1.0])

    # p(delta_beta) ~ Normal_1(mu, sigma_sq)
    delta_beta_dist = normal_v
    delta_beta_mu = np.array([1.0])
    delta_beta_sigma_sq = np.array([1.0])

    # p(delta_beta) ~ Normal_1(mu, sigma_sq)
    delta_gamma_dist = normal_v
    delta_gamma_mu = np.array([1.0])
    delta_gamma_sigma_sq = np.array([1.0])

    #
    # Create prior container
    #

    d = dict()

    if not is_fixed.phi:
        d['phi_alpha'] = phi_alpha
        d['phi_eta'] = np.zeros((dim_j, M))
        d['phi_a'] = np.zeros(M)
        for m in range(M):
            d['phi_eta'][:, m] = phi_dist.map_from_alpha_to_eta(
                alpha=phi_alpha[:, m]
            )
            d['phi_a'][m] = phi_dist.a(eta=d['phi_eta'][:, m])

    if not is_fixed.tau_alpha:
        d['tau_alpha_alpha'] = tau_alpha_alpha
        d['tau_alpha_beta'] = tau_alpha_beta
        d['tau_alpha_eta'] = tau_alpha_dist.map_from_alpha_beta_to_eta(
            alpha=tau_alpha_alpha,
            beta=tau_alpha_beta,
        )
        d['tau_alpha_a'] = tau_alpha_dist.a(eta=d['tau_alpha_eta'])

    if not is_fixed.mu_kappa:
        d['mu_kappa_mu'] = mu_kappa_mu
        d['mu_kappa_sigma'] = mu_kappa_sigma
        d['mu_kappa_eta'] = mu_kappa_dist.map_from_mu_cov_to_eta(
            mu=mu_kappa_mu,
            cov=mu_kappa_sigma,
        )
        d['mu_kappa_a'] = mu_kappa_dist.a(eta=d['mu_kappa_eta'])

    if not is_fixed.lambda_kappa:
        d['lambda_kappa_n'] = lambda_kappa_n
        d['lambda_kappa_v'] = lambda_kappa_v
        d['lambda_kappa_eta'] = lambda_kappa_dist.map_from_n_v_to_eta(
            n=lambda_kappa_n,
            v=lambda_kappa_v,
        )
        d['lambda_kappa_a'] = lambda_kappa_dist.a(eta=d['lambda_kappa_eta'])

    if not is_fixed.beta:
        d['beta_mu'] = beta_mu
        d['beta_sigma'] = beta_sigma
        d['beta_lambda'] = np.linalg.inv(beta_sigma)
        d['beta_eta'] = beta_dist.map_from_mu_cov_to_eta(
            mu=beta_mu,
            cov=beta_sigma,
        )
        d['beta_a'] = beta_dist.a(eta=d['beta_eta'])

    if not is_fixed.gamma:
        d['gamma_mu'] = gamma_mu
        d['gamma_sigma'] = gamma_sigma
        d['gamma_lambda'] = np.linalg.inv(gamma_sigma)
        d['gamma_eta'] = gamma_dist.map_from_mu_cov_to_eta(
            mu=gamma_mu,
            cov=gamma_sigma,
        )
        d['gamma_a'] = gamma_dist.a(eta=d['gamma_eta'])

    if not is_fixed.rho:
        d['rho_mu'] = rho_mu
        d['rho_sigma'] = rho_sigma
        d['rho_lambda'] = np.linalg.inv(rho_sigma)
        d['rho_eta'] = rho_dist.map_from_mu_cov_to_eta(
            mu=rho_mu,
            cov=rho_sigma,
        )
        d['rho_a'] = rho_dist.a(eta=d['rho_eta'])

    if not is_fixed.delta:
        d['delta_mu'] = delta_mu
        d['delta_sigma_sq'] = delta_sigma_sq
        d['delta_eta'] = delta_dist.map_from_mu_sigma_sq_to_eta(
            mu=delta_mu,
            sigma_sq=delta_sigma_sq,
        )
        d['delta_a'] = delta_dist.a(eta=d['delta_eta'])

    if not is_fixed.delta_kappa:
        d['delta_kappa_mu'] = delta_kappa_mu
        d['delta_kappa_sigma_sq'] = delta_kappa_sigma_sq
        d['delta_kappa_eta'] = delta_kappa_dist.map_from_mu_sigma_sq_to_eta(
            mu=delta_kappa_mu,
            sigma_sq=delta_kappa_sigma_sq,
        )
        d['delta_kappa_a'] = delta_kappa_dist.a(eta=d['delta_kappa_eta'])

    if not is_fixed.delta_beta:
        d['delta_beta_mu'] = delta_beta_mu
        d['delta_beta_sigma_sq'] = delta_beta_sigma_sq
        d['delta_beta_eta'] = delta_beta_dist.map_from_mu_sigma_sq_to_eta(
            mu=delta_beta_mu,
            sigma_sq=delta_beta_sigma_sq,
        )
        d['delta_beta_a'] = delta_beta_dist.a(eta=d['delta_beta_eta'])

    if not is_fixed.delta_gamma:
        d['delta_gamma_mu'] = delta_gamma_mu
        d['delta_gamma_sigma_sq'] = delta_gamma_sigma_sq
        d['delta_gamma_eta'] = delta_gamma_dist.map_from_mu_sigma_sq_to_eta(
            mu=delta_gamma_mu,
            sigma_sq=delta_gamma_sigma_sq,
        )
        d['delta_gamma_a'] = delta_gamma_dist.a(eta=d['delta_gamma_eta'])

    prior = Prior(**d)

    return prior
