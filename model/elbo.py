#
# Description:
#   Contains the ELBO structure for the ULSDPB model.
#
# Functions:
#   create_dataset: creates a dataset for the ULSDPB model using
#   the (y_fused_ibn, x, w)-data as input.


# Standard library modules
from collections import namedtuple

# External modules
import numpy as np


ELBO_Container = namedtuple(
    'ELBO_Container',
    [
        'total',
        # y
        'ev_q_log_p_y',
        # z
        'ev_q_log_p_z',
        'entropy_q_z',
        # alpha
        'ev_q_log_p_alpha',
        'entropy_q_alpha',
        # kappa
        'ev_q_log_p_kappa',
        'entropy_q_kappa',
        # phi
        'neg_kl_q_p_phi',
        # tau_alpha
        'neg_kl_q_p_tau_alpha',
        # mu_kappa
        'neg_kl_q_p_mu_kappa',
        # lambda_kappa
        'neg_kl_q_p_lambda_kappa',
        # beta
        'neg_kl_q_p_beta',
        # gamma
        'neg_kl_q_p_gamma',
        # rho
        'neg_kl_q_p_rho',
        # delta
        'neg_kl_q_p_delta',
        # delta_kappa
        'neg_kl_q_p_delta_kappa',
        # delta_beta
        'neg_kl_q_p_delta_beta',
        # delta_gamma
        'neg_kl_q_p_delta_gamma',
    ]
)

LOG_2PI = np.log(2*np.pi)


def compute_elbo_container(
        q,
        M,
        n_purchases_per_basket,
        total_customers,
        total_baskets,
):
    # y
    ev_q_log_p_y = np.sum(q.log_phi * q.counts_phi)

    # z
    ev_q_log_p_z = (
        np.sum(q.counts_basket * q.mu_q_alpha)
        -
        n_purchases_per_basket @ q.log_theta_denom_approx
    )
    entropy_q_z = np.sum(q.entropy_q_z)

    # alpha
    ev_q_log_p_alpha = 0.5 * (
        - total_baskets * M * LOG_2PI
        + total_baskets * np.sum(q.log_tau_alpha)
        - q.tau_alpha @ q.sum_ib_eps_alpha_sq
    )
    entropy_q_alpha = np.sum(q.entropy_q_alpha)

    # kappa
    sum_i_ev_q_kappa = np.sum(q.kappa, axis=0)
    sum_i_ev_q_eps_kappa_outer = (
        total_customers * q.mu_kappa_outer
        + np.sum(q.kappa_outer, axis=0)
        - np.outer(sum_i_ev_q_kappa, q.mu_kappa)
        - np.outer(q.mu_kappa, sum_i_ev_q_kappa)
    )

    ev_q_log_p_kappa = 0.5 * (
        - total_customers * M * LOG_2PI
        + total_customers * q.log_det_lambda_kappa
        - np.sum(q.lambda_kappa * sum_i_ev_q_eps_kappa_outer)
    )
    entropy_q_kappa = np.sum(q.entropy_q_kappa)

    # phi
    ev_q_log_p_minus_log_q_phi = np.sum(q.negative_kl_q_p_phi)

    # tau_alpha
    ev_q_log_p_minus_log_q_tau_alpha = q.negative_kl_q_p_tau_alpha[()]

    # mu_kappa
    ev_q_log_p_minus_log_q_mu_kappa = q.negative_kl_q_p_mu_kappa[()]

    # lambda_kappa
    ev_q_log_p_minus_log_q_lambda_kappa = q.negative_kl_q_p_lambda_kappa[()]

    # beta
    ev_q_log_p_minus_log_q_beta = np.sum(q.negative_kl_q_p_beta)

    # gamma
    ev_q_log_p_minus_log_q_gamma = np.sum(q.negative_kl_q_p_gamma)

    # rho
    ev_q_log_p_minus_log_q_rho = np.sum(q.negative_kl_q_p_rho)

    # delta
    ev_q_log_p_minus_log_q_delta = q.negative_kl_q_p_delta[()]

    # delta_kappa
    ev_q_log_p_minus_log_q_delta_kappa = q.negative_kl_q_p_delta_kappa[()]

    # delta_beta
    ev_q_log_p_minus_log_q_delta_beta = q.negative_kl_q_p_delta_beta[()]

    # delta_gamma
    ev_q_log_p_minus_log_q_delta_gamma = q.negative_kl_q_p_delta_gamma[()]

    total_elbo = (
        ev_q_log_p_y
        + ev_q_log_p_z + entropy_q_z
        + ev_q_log_p_alpha + entropy_q_alpha
        + ev_q_log_p_kappa + entropy_q_kappa
        + ev_q_log_p_minus_log_q_phi
        + ev_q_log_p_minus_log_q_tau_alpha
        + ev_q_log_p_minus_log_q_mu_kappa
        + ev_q_log_p_minus_log_q_lambda_kappa
        + ev_q_log_p_minus_log_q_beta
        + ev_q_log_p_minus_log_q_gamma
        + ev_q_log_p_minus_log_q_rho
        + ev_q_log_p_minus_log_q_delta
        + ev_q_log_p_minus_log_q_delta_kappa
        + ev_q_log_p_minus_log_q_delta_beta
        + ev_q_log_p_minus_log_q_delta_gamma
    )

    return ELBO_Container(
        total=total_elbo,
        # y
        ev_q_log_p_y=ev_q_log_p_y,
        # z
        ev_q_log_p_z=ev_q_log_p_z,
        entropy_q_z=entropy_q_z,
        # alpha
        ev_q_log_p_alpha=ev_q_log_p_alpha,
        entropy_q_alpha=entropy_q_alpha,
        # kappa
        ev_q_log_p_kappa=ev_q_log_p_kappa,
        entropy_q_kappa=entropy_q_kappa,
        # phi
        neg_kl_q_p_phi=ev_q_log_p_minus_log_q_phi,
        # tau_alpha
        neg_kl_q_p_tau_alpha=ev_q_log_p_minus_log_q_tau_alpha,
        # mu_kappa
        neg_kl_q_p_mu_kappa=ev_q_log_p_minus_log_q_mu_kappa,
        # lambda_kappa
        neg_kl_q_p_lambda_kappa=ev_q_log_p_minus_log_q_lambda_kappa,
        # beta
        neg_kl_q_p_beta=ev_q_log_p_minus_log_q_beta,
        # gamma
        neg_kl_q_p_gamma=ev_q_log_p_minus_log_q_gamma,
        # rho
        neg_kl_q_p_rho=ev_q_log_p_minus_log_q_rho,
        # delta
        neg_kl_q_p_delta=ev_q_log_p_minus_log_q_delta,
        # delta_kappa
        neg_kl_q_p_delta_kappa=ev_q_log_p_minus_log_q_delta_kappa,
        # delta_beta
        neg_kl_q_p_delta_beta=ev_q_log_p_minus_log_q_delta_beta,
        # delta_gamma
        neg_kl_q_p_delta_gamma=ev_q_log_p_minus_log_q_delta_gamma,
    )
