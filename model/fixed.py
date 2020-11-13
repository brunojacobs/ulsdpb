#
# Description:
#   Contains the IsFixed and FixedValues structures for the ULSDPB model.
#   A model parameter is only considered fixed if a fixed value is provided.
#
#   For a parameter that is not fixed:
#   - is_fixed.parameter_name: False
#   - fixed_values.parameter_name: None
#
#   For a fixed parameter with variational distribution q:
#   - Variational parameters: 1D/2D numpy arrays filled with np.nan
#   - Variational expectations: as if q is a point mass on the fixed value
#   - Variational entropy: 0.0
#   - Prior parameters: 1D/2D numpy arrays filled with np.nan
#   - Prior log-normalizer (a): 0.0
#
# Functions:
#   create_fixed: creates the IsFixed and FixedValue structures based on the
#   model settings


# Standard library modules
from collections import namedtuple

# External modules
import numpy as np


IsFixed = namedtuple(
    'IsFixed',
    (
        'z',
        'phi',
        'alpha',
        'tau_alpha',
        'kappa',
        'mu_kappa',
        'lambda_kappa',
        'beta',
        'gamma',
        'rho',
        'delta',
        'delta_kappa',
        'delta_beta',
        'delta_gamma',
    ),
)


FixedValues = namedtuple(
    'FixedValues',
    (
        'z_counts_basket',
        'z_counts_phi',
        'phi',
        'alpha',
        'tau_alpha',
        'kappa',
        'mu_kappa',
        'lambda_kappa',
        'beta',
        'gamma',
        'rho',
        'delta',
        'delta_kappa',
        'delta_beta',
        'delta_gamma',
    ),
    defaults=(None,) * 15
)


def create_fixed(
        z_counts_basket=None,
        z_counts_phi=None,
        phi=None,
        alpha=None,
        tau_alpha=None,
        kappa=None,
        mu_kappa=None,
        lambda_kappa=None,
        beta=None,
        gamma=None,
        rho=None,
        delta=None,
        delta_kappa=None,
        delta_beta=None,
        delta_gamma=None,
        emulate_lda_x=False,
        no_regressors=False,
        no_dynamics=False,
        dim_i=None,
        dim_x=None,
        dim_w=None,
        M=None,
):

    if emulate_lda_x:
        assert not no_regressors
        assert not no_dynamics

        """ Emulate the LDA-X model from mksc.2016.0985.
        
        LDA-X works with a single purchase history per customer, instead of
        separate shopping trips per customer. We can emulate its behavior in
        this model by aggregating the shopping trips of a customer into a
        single purchase history and by fixing a few model parameters.
        
        In LDA-X, the motivation probabilities for a customer are distributed:
        
            theta_i | gamma_LDAX, delta_LDAX ~ Dir_M(
                alpha=exp(gamma_LDAX, delta_LDA @ x_i)
            )
        
        with:
            gamma_LDAX[m] ~ Normal, described in mksc.2016.0985
            delta_LDAX[m, k] ~ Normal(mean=0, sigma_sq=0.04)
            x[i] are customer-specific regressors that are standardized
        
        We can emulate this behavior in this model by only focusing on the
        first basket in the model, as later baskets do not exist:
        
            alpha_i1 ~ Normal_M(
                mu = delta + delta_kappa * kappa_i
                    ... + delta_beta * beta @ x_ib
                    ... + delta_gamma * gamma @ w_i,
                sigma_sq = tau_alpha^-1
            )
            theta_i1 = softmax(alpha_i1)

        The LDA-X model can be connected to this set-up as follows:
            delta = gamma_LDAX
            
            gamma = Delta_LDAX
            delta_gamma = 1
            
            kappa = 0-matrix
            delta_kappa = 0
            
            beta = 0-matrix
            delta_beta = 0
        
        This results in:
            alpha_i1 ~ Normal_M(mu=delta + gamma @ w_i, sigma_sq=tau_alpha^-1)
            theta_i1 = softmax(Normal_M(alpha_i1))
         
        The other parts of the model should be fixed as well:
            mu_kappa = 0-vector
            lambda_kappa = 0-matrix
            rho = 0-matrix
        """
        delta_gamma = np.ones(1)

        kappa = np.zeros((dim_i, M))
        delta_kappa = np.zeros(1)

        beta = np.zeros((M, dim_x))
        delta_beta = np.zeros(1)

        mu_kappa = np.zeros(M)
        lambda_kappa = np.zeros((M, M))

        rho = np.zeros((M, M))

    if no_dynamics:
        rho = np.zeros((M, M))

        delta = np.zeros(M)
        delta_kappa = np.ones(1)
        delta_beta = np.ones(1)
        delta_gamma = np.ones(1)

    if no_regressors:
        beta = np.zeros((M, dim_x))
        delta_beta = np.zeros(1)

        gamma = np.zeros((M, dim_w))
        delta_gamma = np.zeros(1)

    is_fixed = IsFixed(
        z=(z_counts_basket is not None) and (z_counts_phi is not None),
        phi=phi is not None,
        alpha=alpha is not None,
        tau_alpha=tau_alpha is not None,
        kappa=kappa is not None,
        mu_kappa=mu_kappa is not None,
        lambda_kappa=lambda_kappa is not None,
        beta=beta is not None,
        gamma=gamma is not None,
        rho=rho is not None,
        delta=delta is not None,
        delta_kappa=delta_kappa is not None,
        delta_beta=delta_beta is not None,
        delta_gamma=delta_gamma is not None,
    )

    fixed_values = FixedValues(
        z_counts_basket=z_counts_basket,
        z_counts_phi=z_counts_phi,
        phi=phi,
        alpha=alpha,
        tau_alpha=tau_alpha,
        kappa=kappa,
        mu_kappa=mu_kappa,
        lambda_kappa=lambda_kappa,
        beta=beta,
        gamma=gamma,
        rho=rho,
        delta=delta,
        delta_kappa=delta_kappa,
        delta_beta=delta_beta,
        delta_gamma=delta_gamma,
    )

    return is_fixed, fixed_values
