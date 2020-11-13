#
# Description:
#   Contains the Stub and State structures for the ULSDPB model.
#
# Functions:
#   create_state: initializes a variational state, based on the state's stub,
#   data, prior, and fixed model parameters.
#
#   check_state: checks the consistency of a variational state


# Standard library modules
from collections import namedtuple

# External modules
import numpy as np

# Own modules
from expfam import normal_v
from expfam import gamma_v
from expfam import dirichlet as dirichlet
from expfam import mvn as mvn
from expfam import wishart as wishart

import model.functions


LOG_2PI_E = np.log(2*np.pi*np.e)


Stub = namedtuple(
    'Stub',
    (
        # variational parameters
        'eta_q_phi',
        'mu_q_alpha',
        'sigma_sq_q_alpha',
        'eta_q_tau_alpha',
        'eta_q_mu_kappa',
        'eta_q_lambda_kappa',
        'eta_q_beta',
        'eta_q_gamma',
        'eta_q_rho',
        'eta_q_delta',
        'eta_q_delta_kappa',
        'eta_q_delta_beta',
        'eta_q_delta_gamma',
        # variational expectations
        'ev_q_counts_basket',
        'ev_q_counts_phi',
        'entropy_q_z',
        'ev_q_kappa',
        'ev_q_kappa_outer',
        # step sizes
        'ss_mu_q_alpha',
        'ss_log_sigma_q_alpha',
    )
)


State = namedtuple(
    'State',
    (
        # z
        'counts_basket',
        'counts_phi',
        'entropy_q_z',
        # phi
        'eta_q_phi',
        'log_phi',
        'negative_kl_q_p_phi',
        # alpha
        'mu_q_alpha',
        'sigma_sq_q_alpha',
        'alpha_sq',
        'log_theta_denom_approx',
        'entropy_q_alpha',
        # tau_alpha
        'eta_q_tau_alpha',
        'log_tau_alpha',
        'tau_alpha',
        'negative_kl_q_p_tau_alpha',
        # kappa
        'kappa',
        'kappa_sq',
        'kappa_outer',
        'entropy_q_kappa',
        # mu_kappa
        'eta_q_mu_kappa',
        'mu_kappa',
        'mu_kappa_outer',
        'negative_kl_q_p_mu_kappa',
        # lambda_kappa
        'eta_q_lambda_kappa',
        'lambda_kappa',
        'log_det_lambda_kappa',
        'negative_kl_q_p_lambda_kappa',
        # beta
        'eta_q_beta',
        'beta',
        'beta_outer',
        'negative_kl_q_p_beta',
        # gamma
        'eta_q_gamma',
        'gamma',
        'gamma_outer',
        'negative_kl_q_p_gamma',
        # rho
        'eta_q_rho',
        'rho',
        'rho_outer',
        'negative_kl_q_p_rho',
        # delta
        'eta_q_delta',
        'delta',
        'delta_sq',
        'negative_kl_q_p_delta',
        # delta_kappa
        'eta_q_delta_kappa',
        'delta_kappa',
        'delta_kappa_sq',
        'negative_kl_q_p_delta_kappa',
        # delta_beta
        'eta_q_delta_beta',
        'delta_beta',
        'delta_beta_sq',
        'negative_kl_q_p_delta_beta',
        # delta_gamma
        'eta_q_delta_gamma',
        'delta_gamma',
        'delta_gamma_sq',
        'negative_kl_q_p_delta_gamma',
        # eps_alpha
        'eps_alpha',
        'sum_ib_eps_alpha_sq',
        # step sizes
        'ss_mu_q_alpha',
        'ss_log_sigma_q_alpha',
    )
)


def create_state(
        state_stub,
        data,
        prior,
        is_fixed,
        fixed_values,
        M,
):

    # z
    if is_fixed.z:
        ev_q_counts_basket = fixed_values.counts_basket
        ev_q_counts_phi = fixed_values.counts_phi
        entropy_q_z = np.zeros(data.total_purchases)
    else:
        ev_q_counts_basket = state_stub.ev_q_counts_basket
        ev_q_counts_phi = state_stub.ev_q_counts_phi
        entropy_q_z = state_stub.entropy_q_z

    # phi
    if is_fixed.phi:
        ev_q_log_phi = np.log(fixed_values.phi)
        negative_kl_q_p_phi = np.zeros(M)
    else:
        ev_q_log_phi = np.zeros((data.dim_j, M))
        negative_kl_q_p_phi = np.zeros(M)

        for m in range(M):
            ev_q_log_phi[:, m] = dirichlet.ev_t(eta=state_stub.eta_q_phi[:, m])

            negative_kl_q_p_phi[m] = -dirichlet.kl_divergence(
                eta_q=state_stub.eta_q_phi[:, m],
                eta_p=prior.phi_eta[:, m],
            )

    # alpha
    mu_q_alpha = np.zeros((data.total_baskets, M))
    sigma_sq_q_alpha = np.zeros((data.total_baskets, M))
    ev_q_alpha_sq = np.zeros((data.total_baskets, M))
    ev_q_log_theta_denom_approx = np.zeros(data.total_baskets)
    entropy_q_alpha = np.zeros(data.total_baskets)

    for ib in range(data.total_baskets):
        if is_fixed.alpha:
            mu_q_alpha[ib] = fixed_values.alpha[ib]
            sigma_sq_q_alpha[ib] = np.zeros(M)
            ev_q_alpha_sq[ib] = mu_q_alpha[ib]**2
            ev_q_log_theta_denom_approx[ib] = (
                model.functions.ev_q_log_theta_denom_ji(
                    mu_q=mu_q_alpha[ib],
                    sigma_sq_q=sigma_sq_q_alpha[ib],
                )
            )
            entropy_q_alpha[ib] = 0.0
        else:
            mu_q_alpha[ib] = state_stub.mu_q_alpha[ib]
            sigma_sq_q_alpha[ib] = state_stub.sigma_sq_q_alpha[ib]
            ev_q_alpha_sq[ib] = (
                state_stub.sigma_sq_q_alpha[ib] + state_stub.mu_q_alpha[ib]**2
            )
            ev_q_log_theta_denom_approx[ib] = (
                model.functions.ev_q_log_theta_denom_ji(
                    mu_q=mu_q_alpha[ib],
                    sigma_sq_q=sigma_sq_q_alpha[ib],
                )
            )
            entropy_q_alpha[ib] = 0.5 * (
                M * LOG_2PI_E + np.sum(np.log(state_stub.sigma_sq_q_alpha[ib]))
            )

    # tau_alpha
    if is_fixed.tau_alpha:
        ev_q_tau_alpha = fixed_values.tau_alpha
        ev_q_log_tau_alpha = np.log(ev_q_tau_alpha)
        negative_kl_q_p_tau_alpha = 0.0
    else:
        ev_q_tau_alpha = gamma_v.ev_x(eta=state_stub.eta_q_tau_alpha)
        ev_q_log_tau_alpha = gamma_v.ev_log_x(eta=state_stub.eta_q_tau_alpha)
        negative_kl_q_p_tau_alpha = -gamma_v.kl_divergence(
            eta_q=state_stub.eta_q_tau_alpha,
            eta_p=prior.tau_alpha_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    negative_kl_q_p_tau_alpha = np.array(negative_kl_q_p_tau_alpha)

    # kappa
    ev_q_kappa = np.zeros((data.dim_i, M))
    ev_q_kappa_outer = np.zeros((data.dim_i, M, M))
    ev_q_kappa_sq = np.zeros((data.dim_i, M))
    entropy_q_kappa = np.zeros(data.dim_i)

    if is_fixed.kappa:
        for i in range(data.dim_i):
            ev_q_kappa[i] = fixed_values.kappa[i]
            ev_q_kappa_outer[i] = np.outer(ev_q_kappa[i], ev_q_kappa[i])
            ev_q_kappa_sq[i] = np.diag(ev_q_kappa_outer[i])
            entropy_q_kappa[i] = 0.0
    else:
        for i in range(data.dim_i):
            ev_q_kappa[i] = state_stub.ev_q_kappa[i]
            ev_q_kappa_outer[i] = state_stub.ev_q_kappa_outer[i]
            cov_q_kappa_i = (
                ev_q_kappa_outer[i] - np.outer(ev_q_kappa[i], ev_q_kappa[i])
            )
            ev_q_kappa_sq[i] = np.diag(ev_q_kappa_outer[i])
            entropy_q_kappa[i] = 0.5 * (
                M * LOG_2PI_E + np.linalg.slogdet(cov_q_kappa_i)[1]
            )

    # mu_kappa
    if is_fixed.mu_kappa:
        ev_q_mu_kappa = fixed_values.mu_kappa
        ev_q_mu_kappa_outer = np.outer(ev_q_mu_kappa, ev_q_mu_kappa)
        negative_kl_q_p_mu_kappa = 0.0
    else:
        ev_q_mu_kappa = mvn.ev_x(eta=state_stub.eta_q_mu_kappa)
        ev_q_mu_kappa_outer = mvn.ev_outer_x(eta=state_stub.eta_q_mu_kappa)
        negative_kl_q_p_mu_kappa = -mvn.kl_divergence(
            eta_q=state_stub.eta_q_mu_kappa,
            eta_p=prior.mu_kappa_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    negative_kl_q_p_mu_kappa = np.array(negative_kl_q_p_mu_kappa)

    # lambda_kappa
    if is_fixed.lambda_kappa:
        ev_q_lambda_kappa = fixed_values.lambda_kappa
        if np.allclose(ev_q_lambda_kappa, 0.0):
            ev_q_log_det_lambda_kappa = 0.0
        else:
            ev_q_log_det_lambda_kappa = np.linalg.slogdet(ev_q_lambda_kappa)[1]
        negative_kl_q_p_lambda_kappa = 0.0
    else:
        ev_q_lambda_kappa = wishart.ev_x(eta=state_stub.eta_q_lambda_kappa)
        ev_q_log_det_lambda_kappa = (
            wishart.ev_log_det_x(eta=state_stub.eta_q_lambda_kappa)
        )
        negative_kl_q_p_lambda_kappa = -wishart.kl_divergence(
            eta_q=state_stub.eta_q_lambda_kappa,
            eta_p=prior.lambda_kappa_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    ev_q_log_det_lambda_kappa = np.array(ev_q_log_det_lambda_kappa)
    negative_kl_q_p_lambda_kappa = np.array(negative_kl_q_p_lambda_kappa)

    # beta
    if is_fixed.beta:
        ev_q_beta = fixed_values.beta
        ev_q_beta_outer = np.zeros((M, data.dim_x, data.dim_x))
        for m in range(M):
            ev_q_beta_outer[m] = np.outer(ev_q_beta[m], ev_q_beta[m])
        negative_kl_q_p_beta = np.zeros(M)
    else:
        ev_q_beta = np.zeros((M, data.dim_x))
        ev_q_beta_outer = np.zeros((M, data.dim_x, data.dim_x))
        negative_kl_q_p_beta = np.zeros(M)
        for m in range(M):
            ev_q_beta[m] = mvn.ev_x(eta=state_stub.eta_q_beta[m])
            ev_q_beta_outer[m] = mvn.ev_outer_x(eta=state_stub.eta_q_beta[m])
            negative_kl_q_p_beta[m] = -mvn.kl_divergence(
                eta_q=state_stub.eta_q_beta[m],
                eta_p=prior.beta_eta,
            )

    # gamma
    if is_fixed.gamma:
        ev_q_gamma = fixed_values.gamma
        ev_q_gamma_outer = np.zeros((M, data.dim_w, data.dim_w))
        for m in range(M):
            ev_q_gamma_outer[m] = np.outer(ev_q_gamma[m], ev_q_gamma[m])
        negative_kl_q_p_gamma = np.zeros(M)
    else:
        ev_q_gamma = np.zeros((M, data.dim_w))
        ev_q_gamma_outer = np.zeros((M, data.dim_w, data.dim_w))
        negative_kl_q_p_gamma = np.zeros(M)
        for m in range(M):
            ev_q_gamma[m] = mvn.ev_x(eta=state_stub.eta_q_gamma[m])
            ev_q_gamma_outer[m] = mvn.ev_outer_x(eta=state_stub.eta_q_gamma[m])
            negative_kl_q_p_gamma[m] = -mvn.kl_divergence(
                eta_q=state_stub.eta_q_gamma[m],
                eta_p=prior.gamma_eta,
            )

    # rho
    if is_fixed.rho:
        ev_q_rho = fixed_values.rho
        ev_q_rho_outer = np.zeros((M, M, M))
        for m in range(M):
            ev_q_rho_outer[m] = np.outer(ev_q_rho[m], ev_q_rho[m])
        negative_kl_q_p_rho = np.zeros(M)
    else:
        ev_q_rho = np.zeros((M, M))
        ev_q_rho_outer = np.zeros((M, M, M))
        negative_kl_q_p_rho = np.zeros(M)
        for m in range(M):
            ev_q_rho[m] = mvn.ev_x(eta=state_stub.eta_q_rho[m])
            ev_q_rho_outer[m] = mvn.ev_outer_x(eta=state_stub.eta_q_rho[m])
            negative_kl_q_p_rho[m] = -mvn.kl_divergence(
                eta_q=state_stub.eta_q_rho[m],
                eta_p=prior.rho_eta,
            )

    # delta
    if is_fixed.delta:
        ev_q_delta = fixed_values.delta
        ev_q_delta_sq = ev_q_delta**2
        negative_kl_q_p_delta = 0.0
    else:
        ev_q_delta = normal_v.ev_x(eta=state_stub.eta_q_delta)
        ev_q_delta_sq = normal_v.ev_x_sq(eta=state_stub.eta_q_delta)
        negative_kl_q_p_delta = -normal_v.kl_divergence(
            eta_q=state_stub.eta_q_delta,
            eta_p=prior.delta_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    negative_kl_q_p_delta = np.array(
        negative_kl_q_p_delta
    ).squeeze()

    # delta_kappa
    if is_fixed.delta_kappa:
        ev_q_delta_kappa = fixed_values.delta_kappa
        ev_q_delta_kappa_sq = ev_q_delta_kappa**2
        negative_kl_q_p_delta_kappa = 0.0
    else:
        ev_q_delta_kappa = normal_v.ev_x(eta=state_stub.eta_q_delta_kappa)
        ev_q_delta_kappa_sq = normal_v.ev_x_sq(eta=state_stub.eta_q_delta_kappa)
        negative_kl_q_p_delta_kappa = -normal_v.kl_divergence(
            eta_q=state_stub.eta_q_delta_kappa,
            eta_p=prior.delta_kappa_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    ev_q_delta_kappa = np.array(ev_q_delta_kappa).squeeze()
    ev_q_delta_kappa_sq = np.array(ev_q_delta_kappa_sq).squeeze()
    negative_kl_q_p_delta_kappa = np.array(
        negative_kl_q_p_delta_kappa
    ).squeeze()

    # delta_beta
    if is_fixed.delta_beta:
        ev_q_delta_beta = fixed_values.delta_beta
        ev_q_delta_beta_sq = ev_q_delta_beta**2
        negative_kl_q_p_delta_beta = 0.0
    else:
        ev_q_delta_beta = normal_v.ev_x(eta=state_stub.eta_q_delta_beta)
        ev_q_delta_beta_sq = normal_v.ev_x_sq(eta=state_stub.eta_q_delta_beta)
        negative_kl_q_p_delta_beta = -normal_v.kl_divergence(
            eta_q=state_stub.eta_q_delta_beta,
            eta_p=prior.delta_beta_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    ev_q_delta_beta = np.array(ev_q_delta_beta).squeeze()
    ev_q_delta_beta_sq = np.array(ev_q_delta_beta_sq).squeeze()
    negative_kl_q_p_delta_beta = np.array(
        negative_kl_q_p_delta_beta
    ).squeeze()

    # delta_gamma
    if is_fixed.delta_gamma:
        ev_q_delta_gamma = fixed_values.delta_gamma
        ev_q_delta_gamma_sq = ev_q_delta_gamma**2
        negative_kl_q_p_delta_gamma = 0.0
    else:
        ev_q_delta_gamma = normal_v.ev_x(eta=state_stub.eta_q_delta_gamma)
        ev_q_delta_gamma_sq = normal_v.ev_x_sq(eta=state_stub.eta_q_delta_gamma)
        negative_kl_q_p_delta_gamma = -normal_v.kl_divergence(
            eta_q=state_stub.eta_q_delta_gamma,
            eta_p=prior.delta_gamma_eta,
        )

    # Numba quirk: Convert shape scalar and (1,)-array to ()-array
    ev_q_delta_gamma = np.array(ev_q_delta_gamma).squeeze()
    ev_q_delta_gamma_sq = np.array(ev_q_delta_gamma_sq).squeeze()
    negative_kl_q_p_delta_gamma = np.array(
        negative_kl_q_p_delta_gamma
    ).squeeze()

    # ev_q_eps_alpha
    ev_q_eps_alpha = np.zeros((data.total_baskets, M))
    model.functions.calc_ev_q_eps_alpha(
        ev_q_eps_alpha=ev_q_eps_alpha,
        mu_q_alpha=mu_q_alpha,
        ev_q_kappa=ev_q_kappa,
        ev_q_beta=ev_q_beta,
        ev_q_gamma=ev_q_gamma,
        ev_q_rho=ev_q_rho,
        ev_q_delta=ev_q_delta,
        ev_q_delta_kappa=ev_q_delta_kappa,
        ev_q_delta_beta=ev_q_delta_beta,
        ev_q_delta_gamma=ev_q_delta_gamma,
        data=data,
    )

    # ev_q_sum_ib_eps_alpha_sq
    ev_q_sum_ib_eps_alpha_sq = np.zeros(M)
    model.functions.calc_ev_q_sum_ib_eps_alpha_sq(
        ev_q_sum_ib_eps_alpha_sq=ev_q_sum_ib_eps_alpha_sq,
        mu_q_alpha=mu_q_alpha,
        sigma_sq_q_alpha=sigma_sq_q_alpha,
        ev_q_alpha_sq=ev_q_alpha_sq,
        ev_q_kappa=ev_q_kappa,
        ev_q_kappa_sq=ev_q_kappa_sq,
        ev_q_beta=ev_q_beta,
        ev_q_beta_outer=ev_q_beta_outer,
        ev_q_gamma=ev_q_gamma,
        ev_q_gamma_outer=ev_q_gamma_outer,
        ev_q_rho=ev_q_rho,
        ev_q_rho_outer=ev_q_rho_outer,
        ev_q_delta=ev_q_delta,
        ev_q_delta_sq=ev_q_delta_sq,
        ev_q_delta_kappa=ev_q_delta_kappa,
        ev_q_delta_kappa_sq=ev_q_delta_kappa_sq,
        ev_q_delta_beta=ev_q_delta_beta,
        ev_q_delta_beta_sq=ev_q_delta_beta_sq,
        ev_q_delta_gamma=ev_q_delta_gamma,
        ev_q_delta_gamma_sq=ev_q_delta_gamma_sq,
        dim_m=M,
        data=data,
    )

    q = State(
        # z
        counts_basket=ev_q_counts_basket,
        counts_phi=ev_q_counts_phi,
        entropy_q_z=entropy_q_z,
        # phi
        eta_q_phi=state_stub.eta_q_phi,
        log_phi=ev_q_log_phi,
        negative_kl_q_p_phi=negative_kl_q_p_phi,
        # alpha
        mu_q_alpha=mu_q_alpha,
        sigma_sq_q_alpha=sigma_sq_q_alpha,
        alpha_sq=ev_q_alpha_sq,
        log_theta_denom_approx=ev_q_log_theta_denom_approx,
        entropy_q_alpha=entropy_q_alpha,
        # kappa
        kappa=ev_q_kappa,
        kappa_sq=ev_q_kappa_sq,
        kappa_outer=ev_q_kappa_outer,
        entropy_q_kappa=entropy_q_kappa,
        # tau_alpha
        eta_q_tau_alpha=state_stub.eta_q_tau_alpha,
        log_tau_alpha=ev_q_log_tau_alpha,
        tau_alpha=ev_q_tau_alpha,
        negative_kl_q_p_tau_alpha=negative_kl_q_p_tau_alpha,
        # mu_kappa
        eta_q_mu_kappa=state_stub.eta_q_mu_kappa,
        mu_kappa=ev_q_mu_kappa,
        mu_kappa_outer=ev_q_mu_kappa_outer,
        negative_kl_q_p_mu_kappa=negative_kl_q_p_mu_kappa,
        # lambda_kappa
        eta_q_lambda_kappa=state_stub.eta_q_lambda_kappa,
        lambda_kappa=ev_q_lambda_kappa,
        log_det_lambda_kappa=ev_q_log_det_lambda_kappa,
        negative_kl_q_p_lambda_kappa=negative_kl_q_p_lambda_kappa,
        # beta
        eta_q_beta=state_stub.eta_q_beta,
        beta=ev_q_beta,
        beta_outer=ev_q_beta_outer,
        negative_kl_q_p_beta=negative_kl_q_p_beta,
        # gamma
        eta_q_gamma=state_stub.eta_q_gamma,
        gamma=ev_q_gamma,
        gamma_outer=ev_q_gamma_outer,
        negative_kl_q_p_gamma=negative_kl_q_p_gamma,
        # rho
        eta_q_rho=state_stub.eta_q_rho,
        rho=ev_q_rho,
        rho_outer=ev_q_rho_outer,
        negative_kl_q_p_rho=negative_kl_q_p_rho,
        # delta
        eta_q_delta=state_stub.eta_q_delta,
        delta=ev_q_delta,
        delta_sq=ev_q_delta_sq,
        negative_kl_q_p_delta=negative_kl_q_p_delta,
        # delta_kappa
        eta_q_delta_kappa=state_stub.eta_q_delta_kappa,
        delta_kappa=ev_q_delta_kappa,
        delta_kappa_sq=ev_q_delta_kappa_sq,
        negative_kl_q_p_delta_kappa=negative_kl_q_p_delta_kappa,
        # delta_beta
        eta_q_delta_beta=state_stub.eta_q_delta_beta,
        delta_beta=ev_q_delta_beta,
        delta_beta_sq=ev_q_delta_beta_sq,
        negative_kl_q_p_delta_beta=negative_kl_q_p_delta_beta,
        # delta_gamma
        eta_q_delta_gamma=state_stub.eta_q_delta_gamma,
        delta_gamma=ev_q_delta_gamma,
        delta_gamma_sq=ev_q_delta_gamma_sq,
        negative_kl_q_p_delta_gamma=negative_kl_q_p_delta_gamma,
        # eps_alpha
        eps_alpha=ev_q_eps_alpha,
        sum_ib_eps_alpha_sq=ev_q_sum_ib_eps_alpha_sq,
        # step sizes
        ss_mu_q_alpha=state_stub.ss_mu_q_alpha,
        ss_log_sigma_q_alpha=state_stub.ss_log_sigma_q_alpha,
    )

    return q


def check_ev_q_eps_alpha(
        q,
        data,
        M,
):

    # ev_q_eps_alpha
    _ev_q_eps_alpha = np.zeros_like(q.eps_alpha)
    model.functions._calc_ev_q_eps_alpha_loop_ib(
        ev_q_eps_alpha=_ev_q_eps_alpha,
        mu_q_alpha=q.mu_q_alpha,
        ev_q_kappa=q.kappa,
        ev_q_beta=q.beta,
        ev_q_gamma=q.gamma,
        ev_q_rho=q.rho,
        ev_q_delta=q.delta,
        ev_q_delta_kappa=q.delta_kappa,
        ev_q_delta_beta=q.delta_beta,
        ev_q_delta_gamma=q.delta_gamma,
        data=data,
    )
    assert np.allclose(q.eps_alpha, _ev_q_eps_alpha)

    _ev_q_eps_alpha = np.zeros_like(q.eps_alpha)
    model.functions._calc_ev_q_eps_alpha_loop_ibm(
        ev_q_eps_alpha=_ev_q_eps_alpha,
        mu_q_alpha=q.mu_q_alpha,
        ev_q_kappa=q.kappa,
        ev_q_beta=q.beta,
        ev_q_gamma=q.gamma,
        ev_q_rho=q.rho,
        ev_q_delta=q.delta,
        ev_q_delta_kappa=q.delta_kappa,
        ev_q_delta_beta=q.delta_beta,
        ev_q_delta_gamma=q.delta_gamma,
        dim_m=M,
        data=data,
    )
    assert np.allclose(q.eps_alpha, _ev_q_eps_alpha)

    # ev_q_sum_ib_eps_alpha_ib_sq
    _ev_q_eps_alpha_sq = np.zeros((data.total_baskets, M))
    model.functions._calc_ev_q_eps_alpha_sq(
        ev_q_eps_alpha_sq=_ev_q_eps_alpha_sq,
        mu_q_alpha=q.mu_q_alpha,
        sigma_sq_q_alpha=q.sigma_sq_q_alpha,
        ev_q_alpha_sq=q.alpha_sq,
        ev_q_kappa=q.kappa,
        ev_q_kappa_sq=q.kappa_sq,
        ev_q_beta=q.beta,
        ev_q_beta_outer=q.beta_outer,
        ev_q_gamma=q.gamma,
        ev_q_gamma_outer=q.gamma_outer,
        ev_q_rho=q.rho,
        ev_q_rho_outer=q.rho_outer,
        ev_q_delta=q.delta,
        ev_q_delta_sq=q.delta_sq,
        ev_q_delta_kappa=q.delta_kappa,
        ev_q_delta_kappa_sq=q.delta_kappa_sq,
        ev_q_delta_beta=q.delta_beta,
        ev_q_delta_beta_sq=q.delta_beta_sq,
        ev_q_delta_gamma=q.delta_gamma,
        ev_q_delta_gamma_sq=q.delta_gamma_sq,
        dim_m=M,
        data=data,
    )
    assert np.allclose(
        q.sum_ib_eps_alpha_sq, np.sum(_ev_q_eps_alpha_sq, axis=0)
    )


def check_state(
        q,
        data,
        prior,
        is_fixed,
        fixed_values,
        M,
):

    # z
    assert q.counts_basket.shape == (data.total_baskets, M)
    assert q.counts_phi.shape == (data.dim_j, M)
    assert q.entropy_q_z.shape == (data.total_purchases,)

    assert np.isclose(np.sum(q.counts_basket), data.total_purchases)
    assert np.allclose(np.sum(q.counts_basket, 1), data.dim_n)

    assert np.isclose(np.sum(q.counts_phi), data.total_purchases)
    assert np.allclose(np.sum(q.counts_phi, 1), data.n_per_product)

    assert np.all(q.counts_basket >= 0.0)
    assert np.all(q.counts_phi >= 0.0)
    assert np.all(q.entropy_q_z >= 0.0)

    if is_fixed.z:
        assert np.allclose(q.counts_basket, fixed_values.counts_basket)
        assert np.allclose(q.counts_phi, fixed_values.counts_phi)
        assert np.allclose(q.sum_b_entropy_q_z, 0.0)

    # phi
    assert q.log_phi.shape == (data.dim_j, M)
    assert q.negative_kl_q_p_phi.shape == (M,)

    if is_fixed.phi:
        assert np.allclose(q.log_phi, np.log(fixed_values.phi))
        assert np.allclose(q.negative_kl_q_p_phi, 0.0)
    else:
        phi_alpha_q = np.zeros_like(q.eta_q_phi)
        for m in range(M):
            assert np.allclose(
                q.log_phi[:, m], dirichlet.ev_t(eta=q.eta_q_phi[:, m])
            )
            phi_alpha_q[:, m] = dirichlet.map_from_eta_to_alpha(
                eta=q.eta_q_phi[:, m]
            )

        phi_alpha_prior = np.zeros_like(prior.phi_eta)
        for m in range(M):
            phi_alpha_prior[:, m] = dirichlet.map_from_eta_to_alpha(
                eta=prior.phi_eta[:, m]
            )

        assert np.isclose(
            np.sum(phi_alpha_prior) + data.total_purchases,
            np.sum(phi_alpha_q)
        )

        for m in range(M):
            assert np.isclose(
                q.negative_kl_q_p_phi[m],
                -dirichlet.kl_divergence(
                    eta_q=q.eta_q_phi[:, m],
                    eta_p=prior.phi_eta[:, m]
                )
            )

    # alpha
    assert q.mu_q_alpha.shape == (data.total_baskets, M)
    assert q.alpha_sq.shape == (data.total_baskets, M)
    assert q.log_theta_denom_approx.shape == (data.total_baskets,)
    assert q.entropy_q_alpha.shape == (data.total_baskets,)

    import model.functions
    for ib in range(data.total_baskets):
        if is_fixed.alpha:
            assert np.allclose(q.mu_q_alpha[ib], fixed_values.alpha[ib])
            assert np.allclose(q.alpha_sq[ib], q.mu_q_alpha[ib] ** 2)
            assert np.isclose(
                q.log_theta_denom_approx[ib],
                model.functions.ev_q_log_theta_denom_ji(
                    mu_q=q.mu_q_alpha[ib],
                    sigma_sq_q=np.zeros(M),
                )
            )
            assert np.allclose(q.entropy_q_alpha[ib], 0.0)
        else:
            eta_q_alpha_ib = normal_v.map_from_mu_sigma_sq_to_eta(
                mu=q.mu_q_alpha[ib],
                sigma_sq=q.sigma_sq_q_alpha[ib],
            )
            assert np.allclose(q.mu_q_alpha[ib], normal_v.ev_x(eta_q_alpha_ib))
            assert np.allclose(q.alpha_sq[ib], normal_v.ev_x_sq(eta_q_alpha_ib))
            assert np.isclose(
                q.log_theta_denom_approx[ib],
                model.functions.ev_q_log_theta_denom_ji(
                    mu_q=q.mu_q_alpha[ib],
                    sigma_sq_q=q.sigma_sq_q_alpha[ib],
                )
            )
            assert np.allclose(
                q.entropy_q_alpha[ib], normal_v.entropy(eta_q_alpha_ib)
            )

    # tau_alpha
    assert q.tau_alpha.shape == (M,)
    assert q.log_tau_alpha.shape == (M,)
    assert np.asarray(q.negative_kl_q_p_tau_alpha).size == 1

    if is_fixed.tau_alpha:
        assert np.allclose(q.tau_alpha, fixed_values.tau_alpha)
        assert np.isclose(q.negative_kl_q_p_tau_alpha, 0.0)
    else:
        assert np.allclose(q.tau_alpha, gamma_v.ev_x(eta=q.eta_q_tau_alpha))
        assert np.allclose(
            q.log_tau_alpha, gamma_v.ev_log_x(eta=q.eta_q_tau_alpha)
        )
        assert np.isclose(
            q.negative_kl_q_p_tau_alpha,
            -gamma_v.kl_divergence(
                eta_q=q.eta_q_tau_alpha, eta_p=prior.tau_alpha_eta
            )
        )

    # kappa
    assert q.kappa.shape == (data.dim_i, M)
    assert q.kappa_sq.shape == (data.dim_i, M)
    assert q.kappa_outer.shape == (data.dim_i, M, M)
    assert q.entropy_q_kappa.shape == (data.dim_i,)

    for i in range(data.dim_i):
        if is_fixed.kappa:
            assert np.allclose(q.kappa[i], fixed_values.kappa[i])
            assert np.allclose(q.kappa_sq[i], q.kappa[i] ** 2)
            assert np.allclose(
                q.kappa_outer[i], np.outer(q.kappa[i], q.kappa[i])
            )
            assert np.isclose(q.entropy_q_kappa[i], 0.0)
        else:
            cov_q_kappa_i = q.kappa_outer[i] - np.outer(q.kappa[i], q.kappa[i])
            eta_q_kappa_i = mvn.map_from_mu_cov_to_eta(
                mu=q.kappa[i],
                cov=cov_q_kappa_i,
            )
            assert np.allclose(q.kappa[i], mvn.ev_x(eta=eta_q_kappa_i))
            assert np.allclose(
                q.kappa_outer[i], mvn.ev_outer_x(eta=eta_q_kappa_i)
            )
            assert np.isclose(
                q.entropy_q_kappa[i], mvn.entropy(eta=eta_q_kappa_i)
            )

    # mu_kappa
    assert q.mu_kappa.shape == (M,)
    assert q.mu_kappa_outer.shape == (M, M)
    assert np.asarray(q.negative_kl_q_p_mu_kappa).size == 1

    if is_fixed.mu_kappa:
        assert np.allclose(q.mu_kappa, fixed_values.mu_kappa)
        assert np.allclose(q.mu_kappa_outer, np.outer(q.mu_kappa, q.mu_kappa))
        assert np.isclose(q.negative_kl_q_p_mu_kappa, 0.0)
    else:
        assert np.allclose(q.mu_kappa, mvn.ev_x(eta=q.eta_q_mu_kappa))
        assert np.allclose(
            q.mu_kappa_outer, mvn.ev_outer_x(eta=q.eta_q_mu_kappa)
        )
        assert np.isclose(
            q.negative_kl_q_p_mu_kappa,
            -mvn.kl_divergence(eta_q=q.eta_q_mu_kappa, eta_p=prior.mu_kappa_eta)
        )

    # lambda_kappa
    assert q.lambda_kappa.shape == (M, M)
    assert np.asarray(q.log_det_lambda_kappa).size == 1
    assert np.asarray(q.negative_kl_q_p_lambda_kappa).size == 1

    if is_fixed.lambda_kappa:
        assert np.allclose(q.lambda_kappa, fixed_values.lambda_kappa)

        if np.allclose(q.lambda_kappa, 0.0):
            assert np.isclose(q.log_det_lambda_kappa, 0.0)
        else:
            assert np.isclose(
                q.log_det_lambda_kappa, np.linalg.slogdet(q.lambda_kappa)[1]
            )

        assert np.isclose(q.negative_kl_q_p_lambda_kappa, 0.0)
    else:
        assert np.allclose(
            q.lambda_kappa, wishart.ev_x(eta=q.eta_q_lambda_kappa)
        )
        assert np.allclose(
            q.log_det_lambda_kappa,
            wishart.ev_log_det_x(eta=q.eta_q_lambda_kappa)
        )
        assert np.isclose(
            q.negative_kl_q_p_lambda_kappa,
            -wishart.kl_divergence(
                eta_q=q.eta_q_lambda_kappa, eta_p=prior.lambda_kappa_eta
            )
        )

    # beta
    assert q.beta.shape == (M, data.dim_x)
    assert q.beta_outer.shape == (M, data.dim_x, data.dim_x)
    assert q.negative_kl_q_p_beta.shape == (M,)

    if is_fixed.beta:
        for m in range(M):
            assert np.allclose(q.beta[m], fixed_values.beta[m])
            assert np.allclose(q.beta_outer[m], np.outer(q.beta[m], q.beta[m]))
        assert np.allclose(q.negative_kl_q_p_beta, 0.0)
    else:
        for m in range(M):
            assert np.allclose(q.beta[m], mvn.ev_x(eta=q.eta_q_beta[m]))
            assert np.allclose(
                q.beta_outer[m], mvn.ev_outer_x(eta=q.eta_q_beta[m])
            )
            assert np.isclose(
                q.negative_kl_q_p_beta[m],
                -mvn.kl_divergence(eta_q=q.eta_q_beta[m], eta_p=prior.beta_eta)
            )

    # gamma
    assert q.gamma.shape == (M, data.dim_w)
    assert q.gamma_outer.shape == (M, data.dim_w, data.dim_w)
    assert q.negative_kl_q_p_gamma.shape == (M,)

    if is_fixed.gamma:
        for m in range(M):
            assert np.allclose(q.gamma[m], fixed_values.gamma[m])
            assert np.allclose(
                q.gamma_outer[m], np.outer(q.gamma[m], q.gamma[m])
            )
        assert np.allclose(q.negative_kl_q_p_gamma, 0.0)
    else:
        for m in range(M):
            assert np.allclose(q.gamma[m], mvn.ev_x(eta=q.eta_q_gamma[m]))
            assert np.allclose(
                q.gamma_outer[m], mvn.ev_outer_x(eta=q.eta_q_gamma[m])
            )
            assert np.isclose(
                q.negative_kl_q_p_gamma[m],
                -mvn.kl_divergence(
                    eta_q=q.eta_q_gamma[m], eta_p=prior.gamma_eta
                )
            )

    # rho
    assert q.rho.shape == (M, M)
    assert q.rho_outer.shape == (M, M, M)
    assert q.negative_kl_q_p_rho.shape == (M,)

    if is_fixed.rho:
        for m in range(M):
            assert np.allclose(q.rho[m], fixed_values.rho[m])
            assert np.allclose(q.rho_outer[m], np.outer(q.rho[m], q.rho[m]))
        assert np.allclose(q.negative_kl_q_p_rho, 0.0)
    else:
        for m in range(M):
            assert np.allclose(q.rho[m], mvn.ev_x(eta=q.eta_q_rho[m]))
            assert np.allclose(
                q.rho_outer[m], mvn.ev_outer_x(eta=q.eta_q_rho[m])
            )
            assert np.allclose(
                q.negative_kl_q_p_rho[m],
                -mvn.kl_divergence(eta_q=q.eta_q_rho[m], eta_p=prior.rho_eta)
            )

    # delta
    assert q.delta.shape == (M,)
    assert q.delta_sq.shape == (M,)
    assert np.asarray(q.negative_kl_q_p_delta).size == 1

    if is_fixed.delta:
        assert np.allclose(q.delta, fixed_values.delta)
        assert np.allclose(q.delta_sq, q.delta ** 2)
        assert np.isclose(q.negative_kl_q_p_delta, 0.0)
    else:
        assert np.allclose(q.delta, normal_v.ev_x(eta=q.eta_q_delta))
        assert np.allclose(q.delta_sq, normal_v.ev_x_sq(eta=q.eta_q_delta))
        assert np.isclose(
            q.negative_kl_q_p_delta,
            -normal_v.kl_divergence(eta_q=q.eta_q_delta, eta_p=prior.delta_eta)
        )

    # delta_kappa
    assert np.asarray(q.delta_kappa).size == 1
    assert np.asarray(q.delta_kappa_sq).size == 1
    assert np.asarray(q.negative_kl_q_p_delta_kappa).size == 1

    if is_fixed.delta_kappa:
        assert np.isclose(q.delta_kappa, fixed_values.delta_kappa)
        assert np.isclose(q.delta_kappa_sq, q.delta_kappa ** 2)
        assert np.isclose(q.negative_kl_q_p_delta_kappa, 0.0)
    else:
        assert np.isclose(q.delta_kappa, normal_v.ev_x(eta=q.eta_q_delta_kappa))
        assert np.isclose(
            q.delta_kappa_sq, normal_v.ev_x_sq(eta=q.eta_q_delta_kappa)
        )
        assert np.isclose(
            q.negative_kl_q_p_delta_kappa,
            -normal_v.kl_divergence(
                eta_q=q.eta_q_delta_kappa, eta_p=prior.delta_kappa_eta
            )
        )

    # delta_beta
    assert np.asarray(q.delta_beta).size == 1
    assert np.asarray(q.delta_beta_sq).size == 1
    assert np.asarray(q.negative_kl_q_p_delta_beta).size == 1

    if is_fixed.delta_beta:
        assert np.isclose(q.delta_beta, fixed_values.delta_beta)
        assert np.isclose(q.delta_beta_sq, q.delta_beta ** 2)
        assert np.isclose(q.negative_kl_q_p_delta_beta, 0.0)
    else:
        assert np.isclose(q.delta_beta, normal_v.ev_x(eta=q.eta_q_delta_beta))
        assert np.isclose(
            q.delta_beta_sq, normal_v.ev_x_sq(eta=q.eta_q_delta_beta)
        )
        assert np.isclose(
            q.negative_kl_q_p_delta_beta,
            -normal_v.kl_divergence(
                eta_q=q.eta_q_delta_beta, eta_p=prior.delta_beta_eta
            )
        )

    # delta_gamma
    assert np.asarray(q.delta_gamma).size == 1
    assert np.asarray(q.delta_gamma_sq).size == 1
    assert np.asarray(q.negative_kl_q_p_delta_gamma).size == 1

    if is_fixed.delta_gamma:
        assert np.isclose(q.delta_gamma, fixed_values.delta_gamma)
        assert np.isclose(q.delta_gamma_sq, q.delta_gamma ** 2)
        assert np.isclose(q.negative_kl_q_p_delta_gamma, 0.0)
    else:
        assert np.isclose(q.delta_gamma, normal_v.ev_x(eta=q.eta_q_delta_gamma))
        assert np.isclose(
            q.delta_gamma_sq, normal_v.ev_x_sq(eta=q.eta_q_delta_gamma)
        )
        assert np.isclose(
            q.negative_kl_q_p_delta_gamma,
            -normal_v.kl_divergence(
                eta_q=q.eta_q_delta_gamma, eta_p=prior.delta_gamma_eta
            )
        )

    check_ev_q_eps_alpha(
        q=q,
        data=data,
        M=M,
    )
