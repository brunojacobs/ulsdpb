#
# Description:
#   Contains functions related to variational inference in the ULSDPB model.

# External modules
import numpy as np
import numba

# Own modules
from expfam import dirichlet
from expfam import gamma_v
from expfam import misc
from expfam import mvn
from expfam import normal_v
from expfam import wishart

from model import settings

LOG_2PI_E = np.log(2*np.pi*np.e)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# eps_alpha # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@numba.jit(**settings.NUMBA_OPTIONS)
def calc_ev_q_eps_alpha(
        ev_q_eps_alpha,
        mu_q_alpha,
        ev_q_kappa,
        ev_q_beta,
        ev_q_gamma,
        ev_q_rho,
        ev_q_delta,
        ev_q_delta_kappa,
        ev_q_delta_beta,
        ev_q_delta_gamma,
        data,
):
    ev_q_eps_alpha[data.ib_first] = mu_q_alpha[data.ib_first] - (
        ev_q_delta
        +
        ev_q_kappa * ev_q_delta_kappa
        +
        (data.x[data.ib_first] @ ev_q_beta.T) * ev_q_delta_beta
        +
        (data.w @ ev_q_gamma.T) * ev_q_delta_gamma
    )

    ev_q_eps_alpha[data.ib_not_first] = - (
        (mu_q_alpha[data.ib_not_last] @ ev_q_rho.T)
        +
        (data.x[data.ib_not_first] @ ev_q_beta.T)
    )

    ib = 0
    for i in range(data.dim_i):
        ev_q_gamma_w_i = ev_q_gamma @ data.w[i]
        for b in range(data.dim_b[i]):
            if b != 0:
                ev_q_eps_alpha[ib] += (
                    mu_q_alpha[ib] - (ev_q_kappa[i] + ev_q_gamma_w_i)
                )
            ib += 1


@numba.jit(**settings.NUMBA_OPTIONS)
def _calc_ev_q_eps_alpha_loop_ib(
        ev_q_eps_alpha,
        mu_q_alpha,
        ev_q_kappa,
        ev_q_beta,
        ev_q_gamma,
        ev_q_rho,
        ev_q_delta,
        ev_q_delta_kappa,
        ev_q_delta_beta,
        ev_q_delta_gamma,
        data,
):

    ib = 0
    for i in range(data.dim_i):
        for b in range(data.dim_b[i]):
            if b == 0:
                mu_ib = (
                    ev_q_delta
                    +
                    ev_q_kappa[i] * ev_q_delta_kappa
                    +
                    ev_q_beta @ data.x[ib] * ev_q_delta_beta
                    +
                    ev_q_gamma @ data.w[i] * ev_q_delta_gamma
                )
            else:
                mu_ib = (
                    ev_q_rho @ mu_q_alpha[ib - 1]
                    +
                    ev_q_kappa[i]
                    +
                    ev_q_beta @ data.x[ib]
                    +
                    ev_q_gamma @ data.w[i]
                )

            ev_q_eps_alpha[ib] = mu_q_alpha[ib] - mu_ib
            ib += 1

    return ev_q_eps_alpha


@numba.jit(**settings.NUMBA_OPTIONS)
def _calc_ev_q_eps_alpha_loop_ibm(
        ev_q_eps_alpha,
        mu_q_alpha,
        ev_q_kappa,
        ev_q_beta,
        ev_q_gamma,
        ev_q_rho,
        ev_q_delta,
        ev_q_delta_kappa,
        ev_q_delta_beta,
        ev_q_delta_gamma,
        dim_m,
        data,
):

    ib = 0
    for i in range(data.dim_i):
        for b in range(data.dim_b[i]):
            for m in range(dim_m):
                if b == 0:
                    mu_ibm = (
                        ev_q_delta[m]
                        +
                        ev_q_kappa[i, m] * ev_q_delta_kappa
                        +
                        np.sum(ev_q_delta_beta * data.x[ib] * ev_q_beta[m])
                        +
                        np.sum(ev_q_delta_gamma * data.w[i] * ev_q_gamma[m])
                    )
                else:
                    mu_ibm = (
                        np.sum(mu_q_alpha[ib - 1] * ev_q_rho[m])
                        +
                        ev_q_kappa[i, m]
                        +
                        np.sum(data.x[ib] * ev_q_beta[m])
                        +
                        np.sum(data.w[i] * ev_q_gamma[m])
                    )

                ev_q_eps_alpha[ib, m] = mu_q_alpha[ib, m] - mu_ibm
            ib += 1

    return ev_q_eps_alpha


@numba.jit(**settings.NUMBA_OPTIONS)
def calc_ev_q_sum_ib_eps_alpha_sq(
        ev_q_sum_ib_eps_alpha_sq,
        mu_q_alpha,
        sigma_sq_q_alpha,
        ev_q_alpha_sq,
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_beta,
        ev_q_beta_outer,
        ev_q_gamma,
        ev_q_gamma_outer,
        ev_q_rho,
        ev_q_rho_outer,
        ev_q_delta,
        ev_q_delta_sq,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        ev_q_delta_beta,
        ev_q_delta_beta_sq,
        ev_q_delta_gamma,
        ev_q_delta_gamma_sq,
        dim_m,
        data,
):

    # Squared terms from: E_q{ \sum_i mu_{i1}^2 }
    sum_first_mu_sq_ib_squared_terms = (
        data.dim_i * ev_q_delta_sq
        +
        np.sum(ev_q_kappa_sq, axis=0) * ev_q_delta_kappa_sq
        +
        np.sum(np.sum(ev_q_beta_outer * data.x_outer_sum_first, axis=2), axis=1) * ev_q_delta_beta_sq
        +
        np.sum(np.sum(ev_q_gamma_outer * data.w_outer_sum_first, axis=2), axis=1) * ev_q_delta_gamma_sq
    )

    # \sum_i \sum_{b : b != last} E_q[alpha_ib alpha_ib^T]
    sum_not_last_ev_q_alpha_outer_lag = (
        np.diag(np.sum(sigma_sq_q_alpha[data.ib_not_last], axis=0))
        +
        mu_q_alpha[data.ib_not_last].T @ mu_q_alpha[data.ib_not_last]
    )

    # Squared terms from: E_q{ \sum_i \sum_{b : b != 0} mu_{ib}^2 }
    sum_not_first_mu_sq_ib_squared_terms = (
        np.sum(np.sum(ev_q_rho_outer * sum_not_last_ev_q_alpha_outer_lag, axis=2), axis=1)
        +
        data.dim_b_min_1.T @ ev_q_kappa_sq
        +
        np.sum(np.sum(ev_q_beta_outer * data.x_outer_sum_not_first, axis=2), axis=1)
        +
        np.sum(np.sum(ev_q_gamma_outer * data.w_outer_sum_not_first, axis=2), axis=1)
    )

    # \sum_ib E_q[alpha_{ib}^2] + second moments from mu_{ib}^2
    ev_q_sum_ib_eps_alpha_sq[:] = (
        np.sum(ev_q_alpha_sq, axis=0)
        +
        sum_first_mu_sq_ib_squared_terms
        +
        sum_not_first_mu_sq_ib_squared_terms
    )

    # Linear terms from: E_q{ \sum_ib mu_{ib}^2 }
    ib = 0
    ev_q_gamma_w_i = np.zeros(dim_m)
    for i in range(data.dim_i):

        ev_q_gamma_w_i[:] = ev_q_gamma @ data.w[i]

        for b in range(data.dim_b[i]):
            if b == 0:
                s1 = ev_q_delta
                s2 = ev_q_kappa[i] * ev_q_delta_kappa
                s3 = ev_q_beta @ data.x[ib] * ev_q_delta_beta
                s4 = ev_q_gamma_w_i * ev_q_delta_gamma
            else:
                s1 = ev_q_rho @ mu_q_alpha[ib - 1]
                s2 = ev_q_kappa[i]
                s3 = ev_q_beta @ data.x[ib]
                s4 = ev_q_gamma_w_i

            ev_q_mu_ib = s1 + s2 + s3 + s4

            ev_q_mu_sq_ib_linear_terms = 2 * (
                s1 * (s2 + s3 + s4)
                +
                s2 * (s3 + s4)
                +
                s3 * s4
            )

            ev_q_sum_ib_eps_alpha_sq += (
                ev_q_mu_sq_ib_linear_terms
                -
                2 * mu_q_alpha[ib] * ev_q_mu_ib
            )

            ib += 1


def _calc_ev_q_eps_alpha_sq(
        ev_q_eps_alpha_sq,
        mu_q_alpha,
        sigma_sq_q_alpha,
        ev_q_alpha_sq,
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_beta,
        ev_q_beta_outer,
        ev_q_gamma,
        ev_q_gamma_outer,
        ev_q_rho,
        ev_q_rho_outer,
        ev_q_delta,
        ev_q_delta_sq,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        ev_q_delta_beta,
        ev_q_delta_beta_sq,
        ev_q_delta_gamma,
        ev_q_delta_gamma_sq,
        dim_m,
        data,
):
    ev_q_eps_alpha_sq[:] = 0.0

    ib = 0
    ev_q_gamma_w_i = np.zeros(dim_m)
    for i in range(data.dim_i):

        ev_q_gamma_w_i[:] = ev_q_gamma @ data.w[i]
        sum_per_m_ev_q_gamma_outer_w_outer = np.zeros(dim_m)
        for m in range(dim_m):
            sum_per_m_ev_q_gamma_outer_w_outer[m] = np.sum(
                ev_q_gamma_outer[m] * data.w_outer[i]
            )

        for b in range(data.dim_b[i]):

            sum_per_m_ev_q_beta_outer_x_ib_outer = np.zeros(dim_m)
            for m in range(dim_m):
                sum_per_m_ev_q_beta_outer_x_ib_outer[m] = np.sum(
                    ev_q_beta_outer[m] * data.x_outer[ib]
                )

            if b == 0:
                ev_q_mu_sq_ib_quadratic_terms = (
                    ev_q_delta_sq
                    +
                    ev_q_kappa_sq[i] * ev_q_delta_kappa_sq
                    +
                    sum_per_m_ev_q_beta_outer_x_ib_outer * ev_q_delta_beta_sq
                    +
                    sum_per_m_ev_q_gamma_outer_w_outer * ev_q_delta_gamma_sq
                )

                s1 = ev_q_delta
                s2 = ev_q_kappa[i] * ev_q_delta_kappa
                s3 = ev_q_beta @ data.x[ib] * ev_q_delta_beta
                s4 = ev_q_gamma_w_i * ev_q_delta_gamma
            else:
                ev_q_alpha_ib_prev_outer = (
                    np.diag(sigma_sq_q_alpha[ib - 1])
                    +
                    np.outer(mu_q_alpha[ib - 1], mu_q_alpha[ib - 1])
                )

                sum_per_m_ev_q_rho_outer_ev_q_alpha_ib_prev_outer = np.zeros(dim_m)
                for m in range(dim_m):
                    sum_per_m_ev_q_rho_outer_ev_q_alpha_ib_prev_outer[m] = np.sum(
                        ev_q_rho_outer[m] * ev_q_alpha_ib_prev_outer
                    )

                ev_q_mu_sq_ib_quadratic_terms = (
                    sum_per_m_ev_q_rho_outer_ev_q_alpha_ib_prev_outer
                    +
                    ev_q_kappa_sq[i]
                    +
                    sum_per_m_ev_q_beta_outer_x_ib_outer
                    +
                    sum_per_m_ev_q_gamma_outer_w_outer
                )

                s1 = ev_q_rho @ mu_q_alpha[ib - 1]
                s2 = ev_q_kappa[i]
                s3 = ev_q_beta @ data.x[ib]
                s4 = ev_q_gamma_w_i

            ev_q_mu_ib = s1 + s2 + s3 + s4

            ev_q_mu_sq_ib_linear_terms = 2 * (
                s1 * (s2 + s3 + s4)
                +
                s2 * (s3 + s4)
                +
                s3 * s4
            )

            ev_q_mu_sq_ib = (
                ev_q_mu_sq_ib_quadratic_terms
                +
                ev_q_mu_sq_ib_linear_terms
            )

            ev_q_eps_alpha_sq[ib] = (
                ev_q_alpha_sq[ib]
                +
                ev_q_mu_sq_ib
                -
                2 * mu_q_alpha[ib] * ev_q_mu_ib
            )

            ib += 1


@numba.jit(**settings.NUMBA_OPTIONS)
def ev_q_log_theta_denom_ji(mu_q, sigma_sq_q):
    """
    Approximate E_q[log theta_denom] by log E_q[theta_denom].

    Note that E_q[log theta_denom] <= log E_q[theta_denom], so this is an
    upperbound. However, in the ELBO this is premultiplied with -1,
    so effectively we create a lowerbound on the ELBO.
    """

    return misc.log_sum_exp(v=mu_q + 0.5 * sigma_sq_q)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # update_q_i  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@numba.jit(**settings.NUMBA_OPTIONS, parallel=True)
def update_q_local(
        # # variables to be updated
        # z
        theta_q_z,
        ev_q_counts_basket,
        ev_q_entropy_q_z,
        ev_q_counts_phi,
        # alpha
        mu_q_alpha,
        sigma_sq_q_alpha,
        ss_mu_q,
        ss_log_sigma_q,
        ev_q_alpha_sq,
        ev_q_log_theta_denom_approx,
        ev_q_entropy_q_alpha,
        # alpha diagnostics
        updated_mu_q,
        updated_sigma_sq_q,
        updated_both,
        # kappa
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_kappa_outer,
        ev_q_entropy_q_kappa,
        # mix
        ev_q_eps_alpha,
        # others
        ev_q_log_phi,
        ev_q_tau_alpha,
        ev_q_mu_kappa,
        ev_q_lambda_kappa,
        ev_q_rho,
        ev_q_rho_outer,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        dim_m,
        # data
        data,
        # settings
        vi_settings,
        is_fixed,
        # efficient inverse
        U_T_mmul_L_inv,
        v,
        log_det_C,
):

    ev_q_lambda_kappa_mmult_ev_q_mu_kappa = ev_q_lambda_kappa @ ev_q_mu_kappa
    ev_q_sum_m_tau_alpha_m_rho_outer_m = np.zeros((dim_m, dim_m))
    for m in range(dim_m):
        ev_q_sum_m_tau_alpha_m_rho_outer_m += (
            ev_q_tau_alpha[m] * ev_q_rho_outer[m]
        )
    ev_q_sum_m_tau_alpha_m_diag_rho_outer_m = np.diag(
        ev_q_sum_m_tau_alpha_m_rho_outer_m
    )

    is_updated_mu_q = np.empty(data.total_baskets, dtype=np.bool_)

    for i in numba.prange(data.dim_i):

        update_q_i(
            i=i,
            # # variables to be updated
            # z
            theta_q_z=theta_q_z,
            ev_q_counts_basket=ev_q_counts_basket,
            ev_q_entropy_q_z=ev_q_entropy_q_z,
            # alpha
            mu_q_alpha=mu_q_alpha,
            sigma_sq_q_alpha=sigma_sq_q_alpha,
            ss_mu_q=ss_mu_q,
            ss_log_sigma_q=ss_log_sigma_q,
            ev_q_alpha_sq=ev_q_alpha_sq,
            ev_q_log_theta_denom_approx=ev_q_log_theta_denom_approx,
            ev_q_entropy_q_alpha=ev_q_entropy_q_alpha,
            # alpha diagnostics
            updated_mu_q=updated_mu_q,
            updated_sigma_sq_q=updated_sigma_sq_q,
            updated_both=updated_both,
            # kappa
            ev_q_kappa=ev_q_kappa,
            ev_q_kappa_sq=ev_q_kappa_sq,
            ev_q_kappa_outer=ev_q_kappa_outer,
            ev_q_entropy_q_kappa=ev_q_entropy_q_kappa,
            # mix
            ev_q_eps_alpha=ev_q_eps_alpha,
            # others
            ev_q_log_phi=ev_q_log_phi,
            ev_q_tau_alpha=ev_q_tau_alpha,
            ev_q_lambda_kappa_mmult_ev_q_mu_kappa=ev_q_lambda_kappa_mmult_ev_q_mu_kappa,
            ev_q_rho=ev_q_rho,
            ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
            ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
            ev_q_delta_kappa=ev_q_delta_kappa,
            ev_q_delta_kappa_sq=ev_q_delta_kappa_sq,
            dim_m=dim_m,
            # data
            data=data,
            # settings
            vi_settings=vi_settings,
            is_fixed=is_fixed,
            # efficient inverse
            U_T_mmul_L_inv=U_T_mmul_L_inv,
            v=v,
            log_det_C=log_det_C,
            is_updated_mu_q=is_updated_mu_q,
        )

    ev_q_counts_phi[:] = 0.0
    for n in range(data.ib_to_ibn_ub[-1]):
        ev_q_counts_phi[data.y[n]] += theta_q_z[n]


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_i(
        i,
        # # variables to be updated
        # z
        theta_q_z,
        ev_q_counts_basket,
        ev_q_entropy_q_z,
        # alpha
        mu_q_alpha,
        sigma_sq_q_alpha,
        ss_mu_q,
        ss_log_sigma_q,
        ev_q_alpha_sq,
        ev_q_log_theta_denom_approx,
        ev_q_entropy_q_alpha,
        # alpha diagnostics
        updated_mu_q,
        updated_sigma_sq_q,
        updated_both,
        # kappa
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_kappa_outer,
        ev_q_entropy_q_kappa,
        # mix
        ev_q_eps_alpha,
        # others
        ev_q_log_phi,
        ev_q_tau_alpha,
        ev_q_lambda_kappa_mmult_ev_q_mu_kappa,
        ev_q_rho,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        dim_m,
        # data
        data,
        # settings
        vi_settings,
        is_fixed,
        # efficient inverse
        U_T_mmul_L_inv,
        v,
        log_det_C,
        is_updated_mu_q,
):

    i_to_ib_lb_i = data.i_to_ib_lb[i]
    i_to_ib_ub_i = data.i_to_ib_ub[i]

    is_updated_mu_q[i_to_ib_lb_i:i_to_ib_ub_i] = False

    for step in range(vi_settings.n_q_i_steps):

        is_first_step = (step == 0)

        for ib in range(i_to_ib_lb_i, i_to_ib_ub_i):

            # q(z_ib) only has to be updated if mu_q_alpha_ib is updated
            if (not is_fixed.z) and (is_first_step or is_updated_mu_q[ib]):

                update_q_z_ib(
                    ib=ib,
                    # variables to be updated
                    theta_q_z=theta_q_z,
                    ev_q_counts_basket=ev_q_counts_basket,
                    ev_q_entropy_q_z=ev_q_entropy_q_z,
                    # others
                    mu_q_alpha=mu_q_alpha,
                    ev_q_log_phi=ev_q_log_phi,
                    # data
                    y=data.y,
                    ib_to_ibn_lb_ib=data.ib_to_ibn_lb[ib],
                    ib_to_ibn_ub_ib=data.ib_to_ibn_ub[ib],
                )

            if not is_fixed.alpha:

                (
                    updated_mu_q_ib,
                    updated_sigma_sq_q_ib,
                ) = update_q_alpha_ib_ji(
                    ib=ib,
                    # variables to be updated
                    mu_q_alpha=mu_q_alpha,
                    sigma_sq_q_alpha=sigma_sq_q_alpha,
                    ss_mu_q=ss_mu_q,
                    ss_log_sigma_q=ss_log_sigma_q,
                    ev_q_alpha_sq=ev_q_alpha_sq,
                    ev_q_log_theta_denom_approx=ev_q_log_theta_denom_approx,
                    ev_q_entropy_q_alpha=ev_q_entropy_q_alpha,
                    ev_q_eps_alpha=ev_q_eps_alpha,
                    # other
                    ev_q_tau_alpha=ev_q_tau_alpha,
                    ev_q_rho=ev_q_rho,
                    ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
                    ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
                    ev_q_counts_basket=ev_q_counts_basket,
                    dim_m=dim_m,
                    dim_n=data.dim_n,
                    ib_not_last=data.ib_not_last,
                    vi_settings=vi_settings,
                    # diagnostics
                    updated_mu_q=updated_mu_q,
                    updated_sigma_sq_q=updated_sigma_sq_q,
                    updated_both=updated_both,
                )

                if updated_mu_q_ib:
                    is_updated_mu_q[ib] = True
                else:
                    is_updated_mu_q[ib] = False

        # q(kappa_i) only has to be updated if at least one mu_q_alpha_ib is updated for this customer
        if (not is_fixed.kappa) and (is_first_step or np.any(is_updated_mu_q[i_to_ib_lb_i:i_to_ib_ub_i])):

            update_q_kappa_i_solution(
                i,
                # variables to be updated
                ev_q_kappa=ev_q_kappa,
                ev_q_kappa_sq=ev_q_kappa_sq,
                ev_q_kappa_outer=ev_q_kappa_outer,
                ev_q_entropy_q_kappa=ev_q_entropy_q_kappa,
                ev_q_eps_alpha=ev_q_eps_alpha,
                # other
                ev_q_lambda_kappa_mmult_ev_q_mu_kappa=ev_q_lambda_kappa_mmult_ev_q_mu_kappa,
                ev_q_delta_kappa=ev_q_delta_kappa,
                ev_q_delta_kappa_sq=ev_q_delta_kappa_sq,
                ev_q_tau_alpha=ev_q_tau_alpha,
                M=dim_m,
                i_to_ib_lb_i=i_to_ib_lb_i,
                i_to_ib_ub_i=i_to_ib_ub_i,
                # efficient inverse
                U_T_mmul_L_inv=U_T_mmul_L_inv,
                v=v,
                log_det_C=log_det_C,
            )


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_z_ib(
        ib,
        # variables to be updated
        theta_q_z,
        ev_q_counts_basket,
        ev_q_entropy_q_z,
        # others
        mu_q_alpha,
        ev_q_log_phi,
        # data
        y,
        ib_to_ibn_lb_ib,
        ib_to_ibn_ub_ib,
):

    for n in range(ib_to_ibn_lb_ib, ib_to_ibn_ub_ib):

        # Update q(z_ibn)
        log_theta_q_z_ibn_nom = mu_q_alpha[ib] + ev_q_log_phi[y[n]]
        log_theta_q_z_ibn_denom = misc.log_sum_exp(log_theta_q_z_ibn_nom)
        theta_q_z[n] = np.exp(log_theta_q_z_ibn_nom - log_theta_q_z_ibn_denom)

        # Update q(z_ibn) caches
        # Note: ev_q_counts_phi is updated outside of this function
        ev_q_entropy_q_z[n] = -np.sum(
            (log_theta_q_z_ibn_nom - log_theta_q_z_ibn_denom) * theta_q_z[n]
        )

    ev_q_counts_basket[ib] = np.sum(
        theta_q_z[ib_to_ibn_lb_ib:ib_to_ibn_ub_ib],
        axis=0,
    )


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_alpha_ib_ji(
        ib,
        # variables to be updated
        mu_q_alpha,
        sigma_sq_q_alpha,
        ss_mu_q,
        ss_log_sigma_q,
        ev_q_alpha_sq,
        ev_q_log_theta_denom_approx,
        ev_q_entropy_q_alpha,
        ev_q_eps_alpha,
        # other
        ev_q_tau_alpha,
        ev_q_rho,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_counts_basket,
        # data
        dim_m,
        dim_n,
        ib_not_last,
        vi_settings,
        # diagnostics
        updated_mu_q,
        updated_sigma_sq_q,
        updated_both,
):
    ev_q_mu_ib = mu_q_alpha[ib] - ev_q_eps_alpha[ib]

    if ib_not_last[ib]:
        cleaned_ev_q_eps_alpha_ib_next = (
            ev_q_eps_alpha[ib + 1] + ev_q_rho @ mu_q_alpha[ib]
        )
    else:
        cleaned_ev_q_eps_alpha_ib_next = np.zeros(dim_m)

    ev_q_tau_cleaned_rho = (
        (ev_q_tau_alpha * cleaned_ev_q_eps_alpha_ib_next) @ ev_q_rho
    )

    pre_update_elbo = elbo_propto_q_alpha_ib_cached_ji(
        mu_q_alpha_ib=mu_q_alpha[ib],
        sigma_sq_q_alpha_ib=sigma_sq_q_alpha[ib],
        ev_q_log_theta_denom_approx_ib=ev_q_log_theta_denom_approx[ib],
        entropy_q_alpha_ib=ev_q_entropy_q_alpha[ib],
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket_ib=ev_q_counts_basket[ib],
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib=dim_n[ib],
        ib_not_last_ib=ib_not_last[ib],
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
    )

    current_elbo = pre_update_elbo

    # Update mu_q_ib
    (
        updated_mu_q_ib,
        current_elbo,
        ev_q_theta_nominator_ib,
    ) = update_q_alpha_ib_ji_mu(
        ib=ib,
        # variables to be updated
        mu_q_alpha=mu_q_alpha,
        ss_mu_q=ss_mu_q,
        ev_q_alpha_sq=ev_q_alpha_sq,
        ev_q_log_theta_denom_approx=ev_q_log_theta_denom_approx,
        ev_q_eps_alpha=ev_q_eps_alpha,
        # other
        sigma_sq_q_alpha=sigma_sq_q_alpha,
        ev_q_entropy_q_alpha=ev_q_entropy_q_alpha,
        cleaned_ev_q_eps_alpha_ib_next=cleaned_ev_q_eps_alpha_ib_next,
        current_elbo=current_elbo,
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket=ev_q_counts_basket,
        ev_q_rho=ev_q_rho,
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
        dim_n=dim_n,
        ib_not_last=ib_not_last,
        vi_settings=vi_settings,
    )

    if updated_mu_q_ib:
        updated_mu_q[ib] += 1

    # Update sigma_sq_q_ib
    (
        updated_sigma_sq_q_ib,
        current_elbo,
    ) = update_q_alpha_ib_ji_sigma_sq(
        ib=ib,
        # variables to be updated
        sigma_sq_q_alpha=sigma_sq_q_alpha,
        ss_log_sigma_q=ss_log_sigma_q,
        ev_q_alpha_sq=ev_q_alpha_sq,
        ev_q_entropy_q_alpha=ev_q_entropy_q_alpha,
        ev_q_log_theta_denom_approx=ev_q_log_theta_denom_approx,
        # other
        mu_q_alpha=mu_q_alpha,
        ev_q_theta_nominator_ib=ev_q_theta_nominator_ib,
        current_elbo=current_elbo,
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket=ev_q_counts_basket,
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
        dim_m=dim_m,
        dim_n=dim_n,
        ib_not_last=ib_not_last,
        vi_settings=vi_settings,
    )

    if updated_sigma_sq_q_ib:
        updated_sigma_sq_q[ib] += 1

    if updated_mu_q_ib and updated_sigma_sq_q_ib:
        updated_both[ib] += 1

    return updated_mu_q_ib, updated_sigma_sq_q_ib


@numba.jit(**settings.NUMBA_OPTIONS)
def elbo_propto_q_alpha_ib_cached_ji(
        mu_q_alpha_ib,
        sigma_sq_q_alpha_ib,
        ev_q_log_theta_denom_approx_ib,
        entropy_q_alpha_ib,
        ev_q_mu_ib,
        ev_q_counts_basket_ib,
        ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib,
        ib_not_last_ib,
        ev_q_tau_cleaned_rho,
):

    ev_q_log_p_alpha_ib_propto_q_alpha_ib = (
        - 0.5 * ev_q_tau_alpha @ (sigma_sq_q_alpha_ib + mu_q_alpha_ib**2)
        +
        (ev_q_tau_alpha * ev_q_mu_ib) @ mu_q_alpha_ib
    )

    ev_q_log_p_z_ib_propto_q_alpha_ib = (
        ev_q_counts_basket_ib @ mu_q_alpha_ib
        -
        n_ib * ev_q_log_theta_denom_approx_ib
    )

    if ib_not_last_ib:
        ev_q_log_p_alpha_ib_next_q_alpha_ib = (
            - 0.5 * ev_q_sum_m_tau_alpha_m_diag_rho_outer_m @ sigma_sq_q_alpha_ib
            - 0.5 * mu_q_alpha_ib.T @ ev_q_sum_m_tau_alpha_m_rho_outer_m @ mu_q_alpha_ib
            + ev_q_tau_cleaned_rho @ mu_q_alpha_ib
        )
    else:
        ev_q_log_p_alpha_ib_next_q_alpha_ib = 0.0

    current_elbo = (
        ev_q_log_p_alpha_ib_propto_q_alpha_ib
        +
        ev_q_log_p_z_ib_propto_q_alpha_ib
        +
        ev_q_log_p_alpha_ib_next_q_alpha_ib
        +
        entropy_q_alpha_ib
    )

    return current_elbo


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_alpha_ib_ji_mu(
        ib,
        # variables to be updated
        mu_q_alpha,
        ss_mu_q,
        ev_q_alpha_sq,
        ev_q_log_theta_denom_approx,
        ev_q_eps_alpha,
        # other
        sigma_sq_q_alpha,
        ev_q_entropy_q_alpha,
        cleaned_ev_q_eps_alpha_ib_next,
        current_elbo,
        ev_q_mu_ib,
        ev_q_counts_basket,
        ev_q_rho,
        ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_tau_cleaned_rho,
        dim_n,
        ib_not_last,
        vi_settings,
):
    # Compute gradient
    ev_q_theta_nominator_ib = np.exp(
        mu_q_alpha[ib] + 0.5 * sigma_sq_q_alpha[ib]
    )

    gradient_mu_q_ib = grad_mu_q_alpha_ib_cached_ji(
        mu_q_alpha_ib=mu_q_alpha[ib],
        ev_q_theta_nominator_ib=ev_q_theta_nominator_ib,
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket_ib=ev_q_counts_basket[ib],
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        n_ib=dim_n[ib],
        is_not_last_ib=ib_not_last[ib],
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
    )

    # Create candidates
    candidate_mu_q = mu_q_alpha[ib] + ss_mu_q[ib] * gradient_mu_q_ib

    candidate_ev_q_log_theta_denom_approx_ib = ev_q_log_theta_denom_ji(
        mu_q=candidate_mu_q,
        sigma_sq_q=sigma_sq_q_alpha[ib],
    )

    candidate_elbo = elbo_propto_q_alpha_ib_cached_ji(
        mu_q_alpha_ib=candidate_mu_q,
        sigma_sq_q_alpha_ib=sigma_sq_q_alpha[ib],
        ev_q_log_theta_denom_approx_ib=candidate_ev_q_log_theta_denom_approx_ib,
        entropy_q_alpha_ib=ev_q_entropy_q_alpha[ib],
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket_ib=ev_q_counts_basket[ib],
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib=dim_n[ib],
        ib_not_last_ib=ib_not_last[ib],
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
    )

    if candidate_elbo - current_elbo > vi_settings.min_elbo_diff:

        updated_mu_q_ib = True

        # Update mu_q_ib
        mu_q_alpha[ib] = candidate_mu_q

        # Update mu_q_ib caches
        ev_q_alpha_sq[ib] = sigma_sq_q_alpha[ib] + candidate_mu_q**2
        ev_q_log_theta_denom_approx[ib] = candidate_ev_q_log_theta_denom_approx_ib

        ev_q_theta_nominator_ib = np.exp(candidate_mu_q + 0.5 * sigma_sq_q_alpha[ib])

        # eps_alpha_ib
        ev_q_eps_alpha[ib] = candidate_mu_q - ev_q_mu_ib

        if ib_not_last[ib]:
            ev_q_eps_alpha[ib + 1] = cleaned_ev_q_eps_alpha_ib_next - ev_q_rho @ candidate_mu_q

        current_elbo = candidate_elbo

        # Increase the step size
        ss_mu_q[ib] = min(
            ss_mu_q[ib] * vi_settings.ss_factor,
            vi_settings.ss_max
        )
    else:

        updated_mu_q_ib = False

        # Decrease the step size
        ss_mu_q[ib] = max(
            ss_mu_q[ib] / vi_settings.ss_factor,
            vi_settings.ss_min
        )

    return updated_mu_q_ib, current_elbo, ev_q_theta_nominator_ib


@numba.jit(**settings.NUMBA_OPTIONS)
def grad_mu_q_alpha_ib_cached_ji(
        mu_q_alpha_ib,
        ev_q_theta_nominator_ib,
        ev_q_mu_ib,
        ev_q_counts_basket_ib,
        ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        n_ib,
        is_not_last_ib,
        ev_q_tau_cleaned_rho,
):
    grad_mu_q_alpha_ib_log_p_alpha_ib = (
        ev_q_tau_alpha * (ev_q_mu_ib - mu_q_alpha_ib)
    )

    grad_mu_q_alpha_ib_log_p_z_ib = (
        ev_q_counts_basket_ib
        -
        n_ib * ev_q_theta_nominator_ib / np.sum(ev_q_theta_nominator_ib)
    )

    if is_not_last_ib:
        grad_mu_q_alpha_ib_log_p_alpha_ib_next = (
            - ev_q_sum_m_tau_alpha_m_rho_outer_m @ mu_q_alpha_ib
            + ev_q_tau_cleaned_rho
        )

        gradient_mu_q_ib = (
            grad_mu_q_alpha_ib_log_p_alpha_ib
            +
            grad_mu_q_alpha_ib_log_p_z_ib
            +
            grad_mu_q_alpha_ib_log_p_alpha_ib_next
        )
    else:
        gradient_mu_q_ib = (
            grad_mu_q_alpha_ib_log_p_alpha_ib
            +
            grad_mu_q_alpha_ib_log_p_z_ib
        )

    return gradient_mu_q_ib


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_alpha_ib_ji_sigma_sq(
        ib,
        # variables to be updated
        sigma_sq_q_alpha,
        ss_log_sigma_q,
        ev_q_alpha_sq,
        ev_q_entropy_q_alpha,
        ev_q_log_theta_denom_approx,
        # other
        mu_q_alpha,
        ev_q_theta_nominator_ib,
        current_elbo,
        ev_q_mu_ib,
        ev_q_counts_basket,
        ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        ev_q_tau_cleaned_rho,
        dim_m,
        dim_n,
        ib_not_last,
        vi_settings,
):

    gradient_sigma_sq_q_ib = grad_sigma_sq_q_alpha_ib_cached_ji(
        sigma_sq_q_alpha_ib=sigma_sq_q_alpha[ib],
        ev_q_theta_nominator_ib=ev_q_theta_nominator_ib,
        ev_q_tau_ib=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib=dim_n[ib],
        is_not_last_ib=ib_not_last[ib],
    )

    gradient_log_sigma_q = (
        2 * sigma_sq_q_alpha[ib] * gradient_sigma_sq_q_ib
    )

    cand_log_sigma_q = (
        0.5 * np.log(sigma_sq_q_alpha[ib])
        +
        ss_log_sigma_q[ib] * gradient_log_sigma_q
    )
    candidate_sigma_sq_q = np.exp(2 * cand_log_sigma_q)

    candidate_entropy_q_alpha_ib = (
        0.5 * dim_m * LOG_2PI_E + np.sum(cand_log_sigma_q)
    )

    candidate_ev_q_log_theta_denom_approx_ib = ev_q_log_theta_denom_ji(
        mu_q=mu_q_alpha[ib],
        sigma_sq_q=candidate_sigma_sq_q,
    )

    candidate_elbo = elbo_propto_q_alpha_ib_cached_ji(
        mu_q_alpha_ib=mu_q_alpha[ib],
        sigma_sq_q_alpha_ib=candidate_sigma_sq_q,
        ev_q_log_theta_denom_approx_ib=candidate_ev_q_log_theta_denom_approx_ib,
        entropy_q_alpha_ib=candidate_entropy_q_alpha_ib,
        ev_q_mu_ib=ev_q_mu_ib,
        ev_q_counts_basket_ib=ev_q_counts_basket[ib],
        ev_q_tau_alpha=ev_q_tau_alpha,
        ev_q_sum_m_tau_alpha_m_rho_outer_m=ev_q_sum_m_tau_alpha_m_rho_outer_m,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m=ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib=dim_n[ib],
        ib_not_last_ib=ib_not_last[ib],
        ev_q_tau_cleaned_rho=ev_q_tau_cleaned_rho,
    )

    if candidate_elbo - current_elbo > vi_settings.min_elbo_diff:

        updated_sigma_sq_q_ib = True

        # Update q(alpha_ib)
        sigma_sq_q_alpha[ib] = candidate_sigma_sq_q

        # Update q(alpha_ib) caches
        ev_q_alpha_sq[ib] = candidate_sigma_sq_q + mu_q_alpha[ib]**2
        ev_q_log_theta_denom_approx[ib] = candidate_ev_q_log_theta_denom_approx_ib
        ev_q_entropy_q_alpha[ib] = candidate_entropy_q_alpha_ib

        current_elbo = candidate_elbo

        # Increase the step size
        ss_log_sigma_q[ib] = min(
            ss_log_sigma_q[ib] * vi_settings.ss_factor,
            vi_settings.ss_max
        )
    else:

        updated_sigma_sq_q_ib = False

        # Decrease the step size
        ss_log_sigma_q[ib] = max(
            ss_log_sigma_q[ib] / vi_settings.ss_factor,
            vi_settings.ss_min
        )

    return updated_sigma_sq_q_ib, current_elbo


@numba.jit(**settings.NUMBA_OPTIONS)
def grad_sigma_sq_q_alpha_ib_cached_ji(
        sigma_sq_q_alpha_ib,
        ev_q_theta_nominator_ib,
        ev_q_tau_ib,
        ev_q_sum_m_tau_alpha_m_diag_rho_outer_m,
        n_ib,
        is_not_last_ib,
):
    grad_sigma_sq_q_alpha_ib_log_p_alpha_ib = -0.5 * ev_q_tau_ib

    grad_sigma_sq_q_alpha_ib_log_p_z_ib = (
        - 0.5 * n_ib * ev_q_theta_nominator_ib / np.sum(ev_q_theta_nominator_ib)
    )

    grad_sigma_sq_q_alpha_ib_entropy_q_alpha_ib = 0.5 * sigma_sq_q_alpha_ib**-1

    if is_not_last_ib:
        grad_sigma_sq_q_alpha_ib_log_p_alpha_ib_next = (
            -0.5 * ev_q_sum_m_tau_alpha_m_diag_rho_outer_m
        )
        gradient_sigma_sq_q_alpha_ib = (
            grad_sigma_sq_q_alpha_ib_log_p_alpha_ib
            +
            grad_sigma_sq_q_alpha_ib_log_p_z_ib
            +
            grad_sigma_sq_q_alpha_ib_entropy_q_alpha_ib
            +
            grad_sigma_sq_q_alpha_ib_log_p_alpha_ib_next
        )
    else:
        gradient_sigma_sq_q_alpha_ib = (
            grad_sigma_sq_q_alpha_ib_log_p_alpha_ib
            +
            grad_sigma_sq_q_alpha_ib_log_p_z_ib
            +
            grad_sigma_sq_q_alpha_ib_entropy_q_alpha_ib
        )

    return gradient_sigma_sq_q_alpha_ib


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_kappa_i_solution(
        i,
        # variables to be updated
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_kappa_outer,
        ev_q_entropy_q_kappa,
        ev_q_eps_alpha,
        # other
        ev_q_lambda_kappa_mmult_ev_q_mu_kappa,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        ev_q_tau_alpha,
        M,
        i_to_ib_lb_i,
        i_to_ib_ub_i,
        # efficient inverse
        U_T_mmul_L_inv,
        v,
        log_det_C,
):
    # Remove q(kappa_i) dependency from eps_alpha_i
    ev_q_eps_alpha[i_to_ib_lb_i] += ev_q_kappa[i] * ev_q_delta_kappa
    ev_q_eps_alpha[(i_to_ib_lb_i + 1):i_to_ib_ub_i] += ev_q_kappa[i]

    ev_q_mb_vector = (
        ev_q_lambda_kappa_mmult_ev_q_mu_kappa
        +
        ev_q_tau_alpha * (
            ev_q_delta_kappa * ev_q_eps_alpha[i_to_ib_lb_i]
            +
            np.sum(ev_q_eps_alpha[(i_to_ib_lb_i + 1):i_to_ib_ub_i], axis=0)
        )
    )

    s_i = ev_q_delta_kappa_sq + i_to_ib_ub_i - i_to_ib_lb_i - 1
    cov_q_i = U_T_mmul_L_inv.T @ np.diag((v + s_i)**-1) @ U_T_mmul_L_inv
    log_det_cov_q_i = -(log_det_C + np.sum(np.log(v + s_i)))

    ev_q_kappa_i = cov_q_i @ ev_q_mb_vector
    ev_q_kappa_outer_i = cov_q_i + np.outer(ev_q_kappa_i, ev_q_kappa_i)
    ev_q_kappa_sq_i = np.diag(ev_q_kappa_outer_i)
    ev_q_entropy_q_kappa_i = 0.5 * (M * LOG_2PI_E + log_det_cov_q_i)

    ev_q_kappa[i] = ev_q_kappa_i
    ev_q_kappa_outer[i] = ev_q_kappa_outer_i
    ev_q_kappa_sq[i] = ev_q_kappa_sq_i
    ev_q_entropy_q_kappa[i] = ev_q_entropy_q_kappa_i

    # Add q(kappa_i) dependency to eps_alpha_i
    ev_q_eps_alpha[i_to_ib_lb_i] -= ev_q_kappa[i] * ev_q_delta_kappa
    ev_q_eps_alpha[(i_to_ib_lb_i + 1):i_to_ib_ub_i] -= ev_q_kappa[i]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # mu_kappa  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_ev_q_eta_p_mu_kappa(
        sum_ev_q_kappa,
        ev_q_lambda_kappa,
        dim_i,
        prior_eta_mu_kappa,
):

    eta_0_from_p_kappa = ev_q_lambda_kappa @ sum_ev_q_kappa

    eta_1_from_p_kappa = -0.5 * dim_i * np.ravel(ev_q_lambda_kappa)

    eta_from_p_kappa = np.concatenate((eta_0_from_p_kappa, eta_1_from_p_kappa))

    return prior_eta_mu_kappa + eta_from_p_kappa


def update_q_mu_kappa(
        eta_q_mu_kappa,
        ev_q_mu_kappa,
        ev_q_mu_kappa_outer,
        ev_q_negative_kl_q_p_mu_kappa,
        sum_ev_q_kappa,
        ev_q_lambda_kappa,
        prior,
        dim_i,
):
    eta_q_mu_kappa[:] = calc_ev_q_eta_p_mu_kappa(
        sum_ev_q_kappa=sum_ev_q_kappa,
        ev_q_lambda_kappa=ev_q_lambda_kappa,
        dim_i=dim_i,
        prior_eta_mu_kappa=prior.mu_kappa_eta,
    )

    ev_q_mu_kappa[:], ev_q_mu_kappa_outer[:] = mvn.split_ev_t(eta=eta_q_mu_kappa)

    ev_q_t = mvn.ev_t(eta=eta_q_mu_kappa)

    ev_q_negative_kl_q_p_mu_kappa[()] = - (
        ev_q_t @ (eta_q_mu_kappa - prior.mu_kappa_eta)
        - mvn.a(eta=eta_q_mu_kappa) + prior.mu_kappa_a
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # lambda_kappa  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_ev_q_eta_p_lambda_kappa(
        sum_ev_q_kappa,
        sum_ev_q_kappa_outer,
        ev_q_mu_kappa,
        ev_q_mu_kappa_outer,
        dim_i,
        prior_eta_lambda_kappa,
):
    eta_0_from_p_kappa = 0.5 * dim_i

    eta_1_from_p_kappa = -0.5 * np.ravel(
        sum_ev_q_kappa_outer +
        dim_i * ev_q_mu_kappa_outer -
        np.outer(sum_ev_q_kappa, ev_q_mu_kappa) -
        np.outer(ev_q_mu_kappa, sum_ev_q_kappa)
    )

    eta_from_p_kappa = np.hstack((eta_0_from_p_kappa, eta_1_from_p_kappa))

    return prior_eta_lambda_kappa + eta_from_p_kappa


def update_q_lambda_kappa(
        eta_q_lambda_kappa,
        ev_q_lambda_kappa,
        ev_q_log_det_lambda_kappa,
        ev_q_negative_kl_q_p_lambda_kappa,
        sum_ev_q_kappa,
        sum_ev_q_kappa_outer,
        ev_q_mu_kappa,
        ev_q_mu_kappa_outer,
        prior,
        dim_i,
):
    eta_q_lambda_kappa[:] = calc_ev_q_eta_p_lambda_kappa(
        sum_ev_q_kappa=sum_ev_q_kappa,
        sum_ev_q_kappa_outer=sum_ev_q_kappa_outer,
        ev_q_mu_kappa=ev_q_mu_kappa,
        ev_q_mu_kappa_outer=ev_q_mu_kappa_outer,
        dim_i=dim_i,
        prior_eta_lambda_kappa=prior.lambda_kappa_eta,
    )

    ev_q_log_det_lambda_kappa[()], ev_q_lambda_kappa[:] = wishart.split_ev_t(
        eta=eta_q_lambda_kappa
    )

    ev_q_t = wishart.ev_t(eta=eta_q_lambda_kappa)

    ev_q_negative_kl_q_p_lambda_kappa[()] = - (
        ev_q_t @ (eta_q_lambda_kappa - prior.lambda_kappa_eta)
        - wishart.a(eta=eta_q_lambda_kappa) + prior.lambda_kappa_a
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # tau_alpha # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_ev_q_eta_p_tau_alpha(
        ev_q_sum_ib_eps_alpha_sq,
        total_baskets,
        dim_m,
        prior_eta_tau_alpha,
):
    eta_0_from_p_alpha = np.full(
        shape=dim_m,
        fill_value=0.5 * total_baskets,
    )
    eta_1_from_p_alpha = -0.5 * ev_q_sum_ib_eps_alpha_sq

    eta_from_p_alpha = np.concatenate((eta_0_from_p_alpha, eta_1_from_p_alpha))

    return prior_eta_tau_alpha + eta_from_p_alpha


def update_q_tau_alpha(
        eta_q_tau_alpha,
        ev_q_tau_alpha,
        ev_q_log_tau_alpha,
        ev_q_negative_kl_q_p_tau_alpha,
        ev_q_sum_ib_eps_alpha_sq,
        prior,
        total_baskets,
        dim_m,
):

    eta_q_tau_alpha[:] = calc_ev_q_eta_p_tau_alpha(
        ev_q_sum_ib_eps_alpha_sq=ev_q_sum_ib_eps_alpha_sq,
        total_baskets=total_baskets,
        dim_m=dim_m,
        prior_eta_tau_alpha=prior.tau_alpha_eta,
    )

    ev_q_t = gamma_v.ev_t(eta=eta_q_tau_alpha)
    ev_q_log_tau_alpha[:] = gamma_v.ev_log_x(eta=eta_q_tau_alpha)
    ev_q_tau_alpha[:] = gamma_v.ev_x(eta=eta_q_tau_alpha)

    ev_q_negative_kl_q_p_tau_alpha[()] = -(
        ev_q_t @ (eta_q_tau_alpha - prior.tau_alpha_eta)
        - gamma_v.a(eta=eta_q_tau_alpha) + prior.tau_alpha_a
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # phi # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def update_q_phi(
        eta_q_phi,
        ev_q_log_phi,
        negative_kl_q_p_phi,
        ev_q_counts_phi,
        prior,
        dim_m,
):
    eta_q_phi[:] = ev_q_counts_phi + prior.phi_eta
    ev_q_log_phi[:] = 0.0
    for m in range(dim_m):
        ev_q_log_phi[:, m] = dirichlet.ev_t(eta_q_phi[:, m])
        negative_kl_q_p_phi[m] = - (
            ev_q_log_phi[:, m] @ (eta_q_phi[:, m] - prior.phi_eta[:, m])
            - dirichlet.a(eta=eta_q_phi[:, m]) + prior.phi_a[m]
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # rho # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@numba.jit(**settings.NUMBA_OPTIONS)
def _update_generic_rho_beta_gamma(
        eta_q_param,
        ev_q_param,
        ev_q_param_outer,
        negative_kl_q_p_param,
        prior_param_eta,
        prior_param_lambda,
        prior_param_a,
        ev_q_XT_X_from_p_alpha_no_tau_alpha,
        ev_q_XT_Y_from_p_alpha_no_tau_alpha,
        ev_q_tau_alpha,
):
    M, K_MVN = ev_q_param.shape

    # log_det_ev_q_XT_X = log_det(ev_q_XT_X_from_p_alpha_no_tau_alpha)
    log_det_ev_q_XT_X = np.linalg.slogdet(ev_q_XT_X_from_p_alpha_no_tau_alpha)[1]
    vec_ev_q_XT_X_from_p_alpha_no_tau_alpha = np.ravel(ev_q_XT_X_from_p_alpha_no_tau_alpha)

    # Efficient inverse for [prior_precision + ev_q_XT_X * ev_q_tau_alpha_m]
    L = np.linalg.cholesky(ev_q_XT_X_from_p_alpha_no_tau_alpha)
    L_inv = np.linalg.inv(L)
    U, v, _ = np.linalg.svd(L_inv @ prior_param_lambda @ L_inv.T)
    U_T_mmul_L_inv = U.T @ L_inv

    for m in range(M):

        eta_0_from_p_alpha_m = ev_q_tau_alpha[m] * ev_q_XT_Y_from_p_alpha_no_tau_alpha[:, m]
        eta_1_from_p_alpha_m = -0.5 * ev_q_tau_alpha[m] * vec_ev_q_XT_X_from_p_alpha_no_tau_alpha
        eta_from_p_alpha_m = np.concatenate((eta_0_from_p_alpha_m, eta_1_from_p_alpha_m))

        eta_q_param[m] = prior_param_eta + eta_from_p_alpha_m

        cov_q_m = U_T_mmul_L_inv.T @ np.diag((v + ev_q_tau_alpha[m])**-1) @ U_T_mmul_L_inv
        mean_q_m = cov_q_m @ eta_q_param[m, :K_MVN]

        ev_q_param[m] = mean_q_m
        ev_q_param_outer[m] = cov_q_m + np.outer(ev_q_param[m], ev_q_param[m])

        log_det_prec_q_m = log_det_ev_q_XT_X + np.sum(np.log(v + ev_q_tau_alpha[m]))

        negative_kl_q_p_param[m] = (
            0.5 * K_MVN
            + ev_q_param[m] @ prior_param_eta[:K_MVN]
            - 0.5 * (np.sum(ev_q_param_outer[m] * prior_param_lambda) + log_det_prec_q_m)
            - prior_param_a
        )


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_beta(
        eta_q_beta,
        ev_q_beta,
        ev_q_beta_outer,
        negative_kl_q_p_beta,
        ev_q_eps_alpha,
        ev_q_tau_alpha,
        ev_q_delta_beta,
        ev_q_delta_beta_sq,
        x,
        x_outer_sum_first,
        x_outer_sum_not_first,
        prior,
        ib_first,
        ib_not_first,
):
    ev_q_eps_alpha[ib_first] += (x[ib_first] @ ev_q_beta.T) * ev_q_delta_beta
    ev_q_eps_alpha[ib_not_first] += x[ib_not_first] @ ev_q_beta.T

    ev_q_XT_X_no_prior_no_tau_alpha = (
        x_outer_sum_first * ev_q_delta_beta_sq
        +
        x_outer_sum_not_first
    )

    ev_q_XT_Y_no_prior_no_tau_alpha = (
        (x[ib_first].T @ ev_q_eps_alpha[ib_first]) * ev_q_delta_beta
        +
        x[ib_not_first].T @ ev_q_eps_alpha[ib_not_first]
    )
    _update_generic_rho_beta_gamma(
        eta_q_param=eta_q_beta,
        ev_q_param=ev_q_beta,
        ev_q_param_outer=ev_q_beta_outer,
        negative_kl_q_p_param=negative_kl_q_p_beta,
        prior_param_eta=prior.beta_eta,
        prior_param_lambda=prior.beta_lambda,
        prior_param_a=prior.beta_a,
        ev_q_XT_X_from_p_alpha_no_tau_alpha=ev_q_XT_X_no_prior_no_tau_alpha,
        ev_q_XT_Y_from_p_alpha_no_tau_alpha=ev_q_XT_Y_no_prior_no_tau_alpha,
        ev_q_tau_alpha=ev_q_tau_alpha,
    )

    ev_q_eps_alpha[ib_first] -= (x[ib_first] @ ev_q_beta.T) * ev_q_delta_beta
    ev_q_eps_alpha[ib_not_first] -= x[ib_not_first] @ ev_q_beta.T


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_gamma(
        eta_q_gamma,
        ev_q_gamma,
        ev_q_gamma_outer,
        negative_kl_q_p_gamma,
        ev_q_eps_alpha,
        ev_q_tau_alpha,
        ev_q_delta_gamma,
        ev_q_delta_gamma_sq,
        w_per_basket,
        w_outer_sum_first,
        w_outer_sum_not_first,
        prior,
        ib_first,
        ib_not_first,
):
    ev_q_eps_alpha[ib_first] += (w_per_basket[ib_first] @ ev_q_gamma.T) * ev_q_delta_gamma
    ev_q_eps_alpha[ib_not_first] += w_per_basket[ib_not_first] @ ev_q_gamma.T

    ev_q_XT_X_from_p_alpha_no_tau_alpha = (
        w_outer_sum_first * ev_q_delta_gamma_sq
        +
        w_outer_sum_not_first
    )

    ev_q_XT_Y_from_p_alpha_no_tau_alpha = (
        (w_per_basket[ib_first].T @ ev_q_eps_alpha[ib_first]) * ev_q_delta_gamma
        +
        w_per_basket[ib_not_first].T @ ev_q_eps_alpha[ib_not_first]
    )

    _update_generic_rho_beta_gamma(
        eta_q_param=eta_q_gamma,
        ev_q_param=ev_q_gamma,
        ev_q_param_outer=ev_q_gamma_outer,
        negative_kl_q_p_param=negative_kl_q_p_gamma,
        prior_param_eta=prior.gamma_eta,
        prior_param_lambda=prior.gamma_lambda,
        prior_param_a=prior.gamma_a,
        ev_q_XT_X_from_p_alpha_no_tau_alpha=ev_q_XT_X_from_p_alpha_no_tau_alpha,
        ev_q_XT_Y_from_p_alpha_no_tau_alpha=ev_q_XT_Y_from_p_alpha_no_tau_alpha,
        ev_q_tau_alpha=ev_q_tau_alpha,
    )

    ev_q_eps_alpha[ib_first] -= (w_per_basket[ib_first] @ ev_q_gamma.T) * ev_q_delta_gamma
    ev_q_eps_alpha[ib_not_first] -= w_per_basket[ib_not_first] @ ev_q_gamma.T


@numba.jit(**settings.NUMBA_OPTIONS)
def update_q_rho(
        eta_q_rho,
        ev_q_rho,
        ev_q_rho_outer,
        negative_kl_q_p_rho,
        ev_q_eps_alpha,
        ev_q_tau_alpha,
        mu_q_alpha,
        sigma_sq_q_alpha,
        prior,
        ib_not_first,
        ib_not_last,
):
    ev_q_eps_alpha[ib_not_first] += mu_q_alpha[ib_not_last] @ ev_q_rho.T

    ev_q_alpha_not_last_outer = (
        np.diag(np.sum(sigma_sq_q_alpha[ib_not_last], axis=0))
        +
        mu_q_alpha[ib_not_last].T @ mu_q_alpha[ib_not_last]
    )
    ev_q_XT_X_from_p_alpha_no_tau_alpha = ev_q_alpha_not_last_outer

    ev_q_XT_Y_from_p_alpha_no_tau_alpha = (
        mu_q_alpha[ib_not_last].T @ ev_q_eps_alpha[ib_not_first]
    )

    _update_generic_rho_beta_gamma(
        eta_q_param=eta_q_rho,
        ev_q_param=ev_q_rho,
        ev_q_param_outer=ev_q_rho_outer,
        negative_kl_q_p_param=negative_kl_q_p_rho,
        prior_param_eta=prior.rho_eta,
        prior_param_lambda=prior.rho_lambda,
        prior_param_a=prior.rho_a,
        ev_q_XT_X_from_p_alpha_no_tau_alpha=ev_q_XT_X_from_p_alpha_no_tau_alpha,
        ev_q_XT_Y_from_p_alpha_no_tau_alpha=ev_q_XT_Y_from_p_alpha_no_tau_alpha,
        ev_q_tau_alpha=ev_q_tau_alpha,
    )

    ev_q_eps_alpha[ib_not_first] -= mu_q_alpha[ib_not_last] @ ev_q_rho.T


def update_q_delta(
        eta_q_delta,
        ev_q_delta,
        ev_q_delta_sq,
        ev_q_negative_kl_q_p_delta,
        ev_q_eps_alpha,
        ev_q_tau_alpha,
        prior,
        dim_i,
        ib_first,
):

    ev_q_eps_alpha[ib_first] += ev_q_delta

    eta_0_from_p_alpha = np.sum(ev_q_eps_alpha[ib_first], axis=0) * ev_q_tau_alpha
    eta_1_from_p_alpha = -0.5 * dim_i * ev_q_tau_alpha
    eta_from_p_alpha = np.concatenate((eta_0_from_p_alpha, eta_1_from_p_alpha))

    eta_q_delta[:] = prior.delta_eta + eta_from_p_alpha

    mu_q, sigma_sq_q = normal_v.map_from_eta_to_mu_sigma_sq(eta=eta_q_delta)

    ev_q_delta[:] = mu_q
    ev_q_delta_sq[:] = sigma_sq_q + mu_q**2

    ev_q_negative_kl_q_p_delta[()] = -normal_v.kl_divergence(
        eta_q=eta_q_delta,
        eta_p=prior.delta_eta,
    )

    ev_q_eps_alpha[ib_first] -= ev_q_delta


def update_q_delta_kappa(
        eta_q_delta_kappa,
        ev_q_delta_kappa,
        ev_q_delta_kappa_sq,
        ev_q_negative_kl_q_p_delta_kappa,
        ev_q_eps_alpha,
        ev_q_kappa,
        ev_q_kappa_sq,
        ev_q_tau_alpha,
        prior,
        ib_first,
):

    ev_q_eps_alpha[ib_first] += ev_q_kappa * ev_q_delta_kappa

    eta_0_from_p_alpha = ev_q_tau_alpha @ (
        np.sum(ev_q_kappa * ev_q_eps_alpha[ib_first], axis=0)
    )
    eta_1_from_p_alpha = -0.5 * np.sum(ev_q_kappa_sq @ ev_q_tau_alpha)
    eta_from_p_alpha = np.array((eta_0_from_p_alpha, eta_1_from_p_alpha))

    eta_q_delta_kappa[:] = prior.delta_kappa_eta + eta_from_p_alpha

    mu_q, sigma_sq_q = normal_v.map_from_eta_to_mu_sigma_sq(eta=eta_q_delta_kappa)

    ev_q_delta_kappa[()] = mu_q
    ev_q_delta_kappa_sq[()] = sigma_sq_q + mu_q**2

    ev_q_negative_kl_q_p_delta_kappa[()] = -normal_v.kl_divergence(
        eta_q=eta_q_delta_kappa,
        eta_p=prior.delta_kappa_eta,
    )

    ev_q_eps_alpha[ib_first] -= ev_q_kappa * ev_q_delta_kappa


def update_q_delta_beta(
        eta_q_delta_beta,
        ev_q_delta_beta,
        ev_q_delta_beta_sq,
        ev_q_negative_kl_q_p_delta_beta,
        ev_q_eps_alpha,
        ev_q_beta,
        ev_q_beta_outer,
        ev_q_tau_alpha,
        dim_m,
        prior,
        x,
        x_outer_sum_first,
        ib_first,
):

    ev_q_eps_alpha[ib_first] += x[ib_first] @ ev_q_beta.T * ev_q_delta_beta

    eta_0_from_p_alpha = 0.0
    eta_1_from_p_alpha = 0.0
    for m in range(dim_m):
        eta_0_from_p_alpha += ev_q_tau_alpha[m] * (
            ev_q_beta[m].T @ x[ib_first].T @ ev_q_eps_alpha[ib_first, m]
        )
        eta_1_from_p_alpha += -0.5 * ev_q_tau_alpha[m] * np.sum(
            ev_q_beta_outer[m] * x_outer_sum_first
        )

    eta_from_p_alpha = np.array((eta_0_from_p_alpha, eta_1_from_p_alpha))

    eta_q_delta_beta[:] = prior.delta_beta_eta + eta_from_p_alpha

    mu_q, sigma_sq_q = normal_v.map_from_eta_to_mu_sigma_sq(
        eta=eta_q_delta_beta
    )

    ev_q_delta_beta[()] = mu_q
    ev_q_delta_beta_sq[()] = sigma_sq_q + mu_q**2

    ev_q_negative_kl_q_p_delta_beta[()] = -normal_v.kl_divergence(
        eta_q=eta_q_delta_beta,
        eta_p=prior.delta_beta_eta,
    )

    ev_q_eps_alpha[ib_first] -= x[ib_first] @ ev_q_beta.T * ev_q_delta_beta


def update_q_delta_gamma(
        eta_q_delta_gamma,
        ev_q_delta_gamma,
        ev_q_delta_gamma_sq,
        ev_q_negative_kl_q_p_delta_gamma,
        ev_q_eps_alpha,
        ev_q_gamma,
        ev_q_gamma_outer,
        ev_q_tau_alpha,
        dim_m,
        prior,
        w,
        w_outer_sum_first,
        ib_first,
):

    ev_q_eps_alpha[ib_first] += w @ ev_q_gamma.T * ev_q_delta_gamma

    eta_0_from_p_alpha = 0.0
    eta_1_from_p_alpha = 0.0
    for m in range(dim_m):
        eta_0_from_p_alpha += ev_q_tau_alpha[m] * (
            ev_q_gamma[m].T @ w.T @ ev_q_eps_alpha[ib_first, m]
        )
        eta_1_from_p_alpha += -0.5 * ev_q_tau_alpha[m] * np.sum(ev_q_gamma_outer[m] * w_outer_sum_first)

    eta_from_p_alpha = np.array((eta_0_from_p_alpha, eta_1_from_p_alpha))

    eta_q_delta_gamma[:] = prior.delta_gamma_eta + eta_from_p_alpha

    mu_q, sigma_sq_q = normal_v.map_from_eta_to_mu_sigma_sq(eta=eta_q_delta_gamma)

    ev_q_delta_gamma[()] = mu_q
    ev_q_delta_gamma_sq[()] = sigma_sq_q + mu_q**2

    ev_q_negative_kl_q_p_delta_gamma[()] = -normal_v.kl_divergence(
        eta_q=eta_q_delta_gamma,
        eta_p=prior.delta_gamma_eta,
    )

    ev_q_eps_alpha[ib_first] -= w @ ev_q_gamma.T * ev_q_delta_gamma
