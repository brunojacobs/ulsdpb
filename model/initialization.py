#
# Description:
#   Contains the initialization logic for the ULSDPB model.
#
# Functions:
#   create_stub_initialization: initializes a stub of the variational state,
#   which is used to construct a complete variational state.


# Standard library modules
import numpy as np

# External modules
import model.state

# Own modules
from expfam import gamma_v
from expfam import mvn
from expfam import normal_v
from expfam import wishart


def create_stub_initialization(
        init_ss_mu_q_alpha_ib,
        init_ss_log_sigma_q_alpha_ib,
        c_jm,
        prior,
        is_fixed,
        data,
        M,
):

    # Initial step sizes for the gradient updates for q(alpha_ib)
    ss_mu_q_alpha = np.full(
        data.total_baskets, init_ss_mu_q_alpha_ib
    )
    ss_log_sigma_q_alpha = np.full(
        data.total_baskets, init_ss_log_sigma_q_alpha_ib
    )

    # Initialization of q(phi) and q(z)

    for j in range(data.dim_j):
        c_jm[j] = data.n_per_product[j] * c_jm[j] / np.sum(c_jm[j])
    c_m_proportions = np.sum(c_jm, axis=0) / np.sum(c_jm)

    # Initialization q(z)
    ev_q_counts_basket_init = (
        data.dim_n[:, np.newaxis] * c_m_proportions[np.newaxis, :]
    )
    ev_q_counts_phi_init = np.zeros((data.dim_j, M))
    ev_q_counts_phi_init[:] = c_jm
    ev_q_entropy_q_z_init = np.zeros(data.total_purchases)
    ev_q_entropy_q_z_init[:] = (
        -np.sum(np.log(c_m_proportions) * c_m_proportions)
    )
    assert np.all(ev_q_entropy_q_z_init >= 0.0)

    # Initialization of q(phi)
    q_phi_alpha_init = np.zeros((data.dim_j, M))
    q_phi_alpha_init[:] = c_jm

    # q(z)
    # If q(z_ibn) is updated before q(alpha_ib):
    # - Initial ev_q_counts_basket has no effect
    # If q(z_ibn) is updated before q(phi):
    # - Initial ev_q_counts_phi has has no effect
    if not is_fixed.z:
        ev_q_counts_basket = ev_q_counts_basket_init
        ev_q_counts_phi = ev_q_counts_phi_init
        entropy_q_z = ev_q_entropy_q_z_init
    else:
        ev_q_counts_basket = None
        ev_q_counts_phi = None
        entropy_q_z = None

    # q(phi)
    if not is_fixed.phi:
        eta_q_phi = c_jm + prior.phi_eta
    else:
        eta_q_phi = None

    # q(alpha)
    if not is_fixed.alpha:
        mu_q_alpha = np.tile(np.zeros(M), (data.total_baskets, 1))
        sigma_sq_q_alpha = np.tile(np.ones(M), (data.total_baskets, 1))
    else:
        mu_q_alpha = None
        sigma_sq_q_alpha = None

    # q(tau_alpha)
    # Only the initial alpha/beta has effect
    if not is_fixed.tau_alpha:
        eta_q_tau_alpha = gamma_v.map_from_alpha_beta_to_eta(
            alpha=prior.tau_alpha_alpha,
            beta=prior.tau_alpha_beta,
        )
    else:
        eta_q_tau_alpha = None

    # q(kappa)
    # If q(kappa_i) is updated before q(delta_kappa) and q(tau_alpha):
    # - Initial covariance/precision of q(kappa_i) has no effect
    if not is_fixed.kappa:
        ev_q_kappa = np.tile(np.zeros(M), (data.dim_i, 1))
        ev_q_kappa_outer = np.tile(np.identity(M), (data.dim_i, 1, 1))
    else:
        ev_q_kappa = None
        ev_q_kappa_outer = None

    # q(mu_kappa)
    # If q(mu_kappa) is updated before q(lambda_kappa):
    # - Initial covariance/precision of q(mu_kappa) has no effect
    if not is_fixed.mu_kappa:
        eta_q_mu_kappa = mvn.map_from_mu_cov_to_eta(
            mu=prior.mu_kappa_mu,
            cov=prior.mu_kappa_sigma,
        )
    else:
        eta_q_mu_kappa = None

    # q(lambda_kappa)
    # Only the initial n*V has effect
    if not is_fixed.lambda_kappa:
        eta_q_lambda_kappa = wishart.map_from_n_v_to_eta(
            n=prior.lambda_kappa_n,
            v=prior.lambda_kappa_v,
        )
    else:
        eta_q_lambda_kappa = None

    # q(beta_m)
    # If q(beta_b) is updated before q(delta_beta) and q(tau_alpha)
    # - Initial covariance/precision of q(beta_m) has no effect
    if not is_fixed.beta:
        eta_q_beta_m = mvn.map_from_mu_cov_to_eta(
            mu=prior.beta_mu,
            cov=prior.beta_sigma,
        )
        eta_q_beta = np.tile(eta_q_beta_m, (M, 1))
    else:
        eta_q_beta = None

    # q(gamma_m)
    # If q(gamma_m) is updated before q(delta_gamma) and q(tau_alpha)
    # - Initial covariance/precision of q(gamma_m) has no effect
    if not is_fixed.gamma:
        eta_q_gamma_m = mvn.map_from_mu_cov_to_eta(
            mu=prior.gamma_mu,
            cov=prior.gamma_sigma,
        )
        eta_q_gamma = np.tile(eta_q_gamma_m, (M, 1))
    else:
        eta_q_gamma = None

    # q(rho_m)
    if not is_fixed.rho:
        eta_q_rho_m = mvn.map_from_mu_cov_to_eta(
            mu=prior.rho_mu,
            cov=prior.rho_sigma / M**2,
        )
        eta_q_rho = np.tile(eta_q_rho_m, (M, 1))
    else:
        eta_q_rho = None

    # q(delta)
    # If q(delta) is updated before q(tau_alpha):
    # - Initial covariance/precision of q(delta) has no effect
    if not is_fixed.delta:
        eta_q_delta = normal_v.map_from_mu_sigma_sq_to_eta(
            mu=prior.delta_mu,
            sigma_sq=prior.delta_sigma_sq,
        )
    else:
        eta_q_delta = None

    # q(delta_kappa)
    if not is_fixed.delta_kappa:
        eta_q_delta_kappa = normal_v.map_from_mu_sigma_sq_to_eta(
            mu=prior.delta_kappa_mu,
            sigma_sq=prior.delta_kappa_sigma_sq / M**2,
        )
    else:
        eta_q_delta_kappa = None

    # q(delta_beta)
    if not is_fixed.delta_beta:
        eta_q_delta_beta = normal_v.map_from_mu_sigma_sq_to_eta(
            mu=prior.delta_beta_mu,
            sigma_sq=prior.delta_beta_sigma_sq / M**2,
        )
    else:
        eta_q_delta_beta = None

    # q(delta_gamma)
    if not is_fixed.delta_gamma:
        eta_q_delta_gamma = normal_v.map_from_mu_sigma_sq_to_eta(
            mu=prior.delta_gamma_mu,
            sigma_sq=prior.delta_gamma_sigma_sq / M**2,
        )
    else:
        eta_q_delta_gamma = None

    state_stub = model.state.Stub(
        # variational parameters
        eta_q_phi=eta_q_phi,
        mu_q_alpha=mu_q_alpha,
        sigma_sq_q_alpha=sigma_sq_q_alpha,
        eta_q_tau_alpha=eta_q_tau_alpha,
        eta_q_mu_kappa=eta_q_mu_kappa,
        eta_q_lambda_kappa=eta_q_lambda_kappa,
        eta_q_beta=eta_q_beta,
        eta_q_gamma=eta_q_gamma,
        eta_q_rho=eta_q_rho,
        eta_q_delta=eta_q_delta,
        eta_q_delta_kappa=eta_q_delta_kappa,
        eta_q_delta_beta=eta_q_delta_beta,
        eta_q_delta_gamma=eta_q_delta_gamma,
        # variational expectations
        ev_q_counts_basket=ev_q_counts_basket,
        ev_q_counts_phi=ev_q_counts_phi,
        entropy_q_z=entropy_q_z,
        ev_q_kappa=ev_q_kappa,
        ev_q_kappa_outer=ev_q_kappa_outer,
        # step sizes
        ss_mu_q_alpha=ss_mu_q_alpha,
        ss_log_sigma_q_alpha=ss_log_sigma_q_alpha,
    )

    return state_stub
