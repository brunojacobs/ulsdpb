# Understanding Large-Scale Dynamic Purchase Behavior

This Python package can be used to estimate the model as proposed in the paper 
**"Understanding Large-Scale Dynamic Purchase Behavior"** by Bruno Jacobs,
Dennis Fok, and Bas Donkers, forthcoming in *Marketing Science*.

- To cite this package, please cite the corresponding publication:

>    Bruno Jacobs, Dennis Fok, Bas Donkers,
>    **"Understanding Large-Scale Dynamic Purchase Behavior"**,
>    *Marketing Science* (forthcoming)

- The most recent version of this package can be found at:
https://brunojacobs.com

This README-file is structured as follows:
- **Step-by-step instructions**: How to run the code
- **Data**: What the input data should look like
- **Output**: Describes the output files and variables

Some minor settings can be configured in the **`settings.py`** file in this
directory. However, changing any of these settings is not required to estimate
the model if this README-file is followed.

The following notation is used in this README-file to denote the dimensions used
in the model:
- *I*: Total number of customers in the data
- *B*: Total number of baskets (shopping trips) in the data
- *J*: Total number of unique products in the data
- *N*: Total number of purchases in the data
- *K_X*: The number of basket-specific predictor variables
- *K_W*: The number of customer-specific predictor variables

## Step-by-step instructions

This package relies on a Python enviroment created using conda, an open-source
package management system. More information about conda can be found at
https://conda.io/. 
 
Use the terminal or an Anaconda Prompt for the following steps:

1. Navigate to the directory that contains this `README.md` file

2. Create the `ulsdpb` conda environment from the `environment.yml` file:

    ```
    conda env create -f environment.yml
    ```

3. Activate the `ulsdpb` environment:

    ```
    conda activate ulsdpb
    ```

    After this step, the terminal prompt should be prefixed with `(ulsdpb)`

4. Initialize pseudocounts for q(phi) using a Collapsed-Gibbs LDA sampler:

    The pseudocounts for q(phi) are initialized using a Collapsed-Gibbs LDA
    sampler by calling the **`initialize_c_jm.py`** script. The following
    arguments are required when calling this script:
    
    - `M`: Number of motivations
    - `S`: Seed for the random number generator of the initialization
    - `N_ITER`: Number of iterations for the Collapsed-Gibbs LDA sampler
    - `N_BURN`: Number of burn-in iterations to be discarded

    This script produces a *J x M* matrix of initial pseudocounts for q(phi),
    written to a .csv file at: `output/M$M/initial_c_jm.csv`, where `$M` is
    replaced by the value for `M`.
    
    For example, to initialize the pseudocounts for 100 motivations, with seed
    0, 10000 iterations, and discarding the first 5000 of these 10000
    iterations, the following command should be used:
    
    ```
    python initialize_c_jm.py -M 100 -S 0 -N_ITER 10000 -N_BURN 5000
    ```
    
    The corresponding output will be written to `output/M100/initial_c_jm.csv`.
    
    Alternatively, this initialization step can be skipped by manually placing
    a .csv file with a *J x M* matrix of initial pseudocounts at the location
    mentioned above. However, do note that this initialization step is specific
    to both the purchase data and the number of motivations in the model

5. Estimate the model using variational inference

    The model can be estimated by calling the **`estimate.py`** script. The
    following arguments are required when calling this script:
    
    - `MODEL`: Model to be estimated. Three values are valid:
        - `FULL`: Complete ULSDPB-model
        - `NO_VAR`: ULSDPB-model without VAR(1) effects
        - `LDA_X`: ULSDPB-model restricted to LDA-X
    - `M`: Number of motivations
    - `N`: Number of iterations for the variational inference algorithm
    - `N_SAVE_PER`: Number of iterations between writing intermediate output
    
    By default, the estimation output is written to the `output/M$M/$MODEL/`
    folder, where `$M$` is replaced by the value for `M` and `$MODEL` by the 
    value for `MODEL`. The output is described in more detail in the Output
    section in this README-file.
    
    For example, to estimate the complete ULSDPB-model, with 100 motivations,
    2000 iterations, and saving the intermediate output after every 500th
    iteration, the following command should be used:

    ```
    python estimate.py -MODEL FULL -M 100 -N_ITER 2000 -N_SAVE_PER 500
    ```
    
    The corresponding output will be written to `output/M100/FULL/`

### Verification of results

To verify that the `ulsdpb` conda environment has been created and activated
correctly, synthetic data files have been included with this package. Summary
estimation results obtained for this data are provided below. These can be used
to verify that the conda environment is set up correctly.

First, the pseudocounts for q(phi) are initialized with `M = 10`:

```
python initialize_c_jm.py -M 10 -S 0 -N_ITER 10000 -N_BURN 5000
```
After the last iteration a log likelihood of `-269041` should be displayed.

Next, each model is estimated using `M = 10`. For each of the three models, the
ELBO-value is provided after completing 100 iterations of the optimzation
algorithm.

For `MODEL = FULL`:
```
python estimate.py -MODEL FULL -M 10 -N_ITER 100 -N_SAVE_PER 100
```
Results in an ELBO value of `-249927.86632913537` after 100 iterations.

For `MODEL = NO_VAR`:
```
python estimate.py -MODEL NO_VAR -M 10 -N_ITER 100 -N_SAVE_PER 100
```
Results in an ELBO value of `-249569.56169985456` after 100 iterations.

For `MODEL = LDA_X`:
```
python estimate.py -MODEL LDA_X -M 10 -N_ITER 100 -N_SAVE_PER 100
```
Results in an ELBO value of `-252986.87489436936` after 100 iterations.

These results are obtained on a MacBook Pro Model A1707, with an
2.9 GHz quad-core Intel Core i7 Kaby Lake (7820HQ) processor.


## Data

The data consists of 3 .csv files: **`y.csv`**, **`x.csv`**, and **`w.csv`**.

The default location to store these files is in the **`data`** folder.


### `y.csv`: The purchase data
- Contains *N* lines with per line, 3 comma separated integers
- Each line represents a tuple of **`customer_id`**, **`basket_id`**, **`product_id`**
- Note: The data should be sorted in ascending order first by `customer_id` and
then by `basket_id`
- **`customer_id`**: ranges from 0 to *I* - 1 (inclusive)
    - Each `customer_id` in {0, 1, ..., *I* - 1} should occur at least once, in
    other words, no `customer_id` value is skipped in the data
- **`basket_id`**: ranges from 0 to *B* - 1 (inclusive)
    - Each `basket_id` in {0, 1, ..., *B* - 1} should occur at least once, in
    other words, no `basket_id` value is skipped in the data
    - `basket_id=0`: The first basket of `customer_id=0`
    - `basket_id=B-1`: The last basket of `customer_id=I-1`
- **`product_id`**: ranges from 0 to *J* - 1 (inclusive)
    - Each `product_id` in {0, 1, ..., *J* - 1} should occur at least once, in
    other words, no `product_id` value is skipped in the data
    - The purchase of product `j` in basket `b` associated with customer `i` is
    represented in the data as the line: `i, b, j`
    - If the purchase quantity of product `j` in `i, b` is greater than one,
    the line `i, b, j` should be duplicated. For example, if product `j` is
    purchased 5 times in `i, b`, the line `i, b, j` should occur 5 times in the
    data

The mapping from customers and products in the data to `customer_id`'s and
`product_id`'s is not important. However, how the baskets in the data are
mapped to a `basket_id` *is important*. The mapping from the baskets of a
customer to `basket_id`'s should be in chronological order for this particular
customer, but the `basket_id`'s do not need to be in chronological order
*across* customers.

This can be ilustrated with a simple example involving two customers:
- `customer_id=0` has five baskets in the data. Therefore, basket_id's={0,1,2,3,4}
should be used for this customer. These basket_id's should be assigned in
chronological order, such that `basket_id=0` refers to the first basket of
`customer_id=0`, `basket_id=1` refers to the second basket of `customer_id=0`, etc.
- `customer_id=1` has three baskets in the data. Therefore, basket_id's={5,6,7}
should be used for this customer. Again, these basket id's should be assigned
in chronological order, such that `basket_id=5` refers to the first basket of
`customer_id=1`, `basket_id=6` refers to the second basket of `customer_id=1`, etc.
- That is, for a given customer_id, the corresponding basket_id's should be
assigned in chronological order. However, between two different customers,
the assigned basket_id's do not have to imply a chronological order. In the
example above, `basket_id=5` (referring to the first basket of `customer_id=1`)
could occur in time before or after any of the basket's for `customer_id=0`
- Keep in mind that the data has to be sorted in ascending order, first by
`customer_id` and then by `basket_id`
 

### `x.csv`: The basket-specific predictor variables
- Contains *B* lines with per line, *K_X* comma separated floats
- The *b*-th line contains the *K_X* predictor variables for basket_id *b*

### `w.csv`: The customer-specific predictor variables
- Contains *I* lines with per line, *K_W* comma separated floats
- The *i*-th line contains the *K_W* predictor variables for customer_id *i* 

## Output

The output consists of several numpy arrays that are saved into a single file
in compressed `.npz` format. The `numpy.load` function should be used to load
these `.npz` files. For more information on this function, refer to the numpy
manual: https://numpy.org/doc/stable/reference/generated/numpy.load.html

Note that any scalar output in a `.npz` file is implicitly converted to a
0D numpy array.
 
### Variational state
The variational state contains parameters of the variational distributions and
expectations of moments under these variational distributions. The variational
state at a particular iteration of the optimization algorithm is written to:

```
output/M$M/$MODEL/state_XXXXXXXXXX.npz
```

where `XXXXXXXXXX` is the 10-digit iteration number with leading zeros. For
example, the output at iteration 1999 would be saved as `state_0000001999.npz`.

For most of the variational distributions, the parameters in the variational
state correspond to the natural parameterizations of these distributions.
Such natural parameters are stored in variables prefixed with `eta_q_`.
Each natural parameter is represented as a vector, stored as a 1D numpy array.
To convert these natural parameters to "regular" parameters, the corresponding
functions in the **`expfam`** folder should be used.

For example, `eta_q_mu_kappa` is the natural parameter vector of q(mu_kappa),
which is an *M*-dimensional multivariate Normal distribution. To convert this 
vector to the distribution's mean-vector and covariance-matrix, the
**`map_from_eta_to_mu_cov`** function from the **`mvn`** module in the
**`expfam`** folder should be used:

```
from expfam import mvn
mean, cov = mvn.map_from_eta_to_mu_cov(eta=eta_q_mu_kappa)
```

Similarly, `eta_q_lambda_kappa` is the natural parameter vector of
q(lambda_kappa), which is an *M*-dimensional Wishart distribution. To convert
this vector to the distribution's degrees of freedom and positive definite scale
matrix, the corresponding function from the **`wishart`** module in the
**`expfam`** folder should be used:

```
from expfam import wishart
n, v = wishart.map_from_eta_to_n_v(eta=eta_q_lambda_kappa)
```

The variational parameters and expectations stored in the variational state are
below grouped and described for each latent variable/parameter separately.

#### Related to q(z_ibn) = Categorical_M
- `counts_basket`: Pseudocounts over the *M* motivations for each basket
    - *B x M* matrix stored as a 2D numpy array
    - The (b, m)-th element is the variational expectation of the number of
    purchases in basket b assigned to motivation m 
- `counts_phi`: Pseudocounts over the *M* motivations for each product
    - *J x M* matrix stored as a 2D numpy array
    - The (j, m)-th element is the variational expectation of the number of
    purchases of product j assigned to motivation m
- `entropy_q_z`: Entropy of each q(z_ibn) distribution
    - *N*-element vector stored as a 1D numpy array
    - The n-th element is the entropy of q(z_ibn)

#### Related to q(phi) = Dirichlet_J
- `eta_q_phi`: Natural variational parameter for each q(phi_m)
    - *J x M* matrix stored as a 2D numpy array
    - The m-th **column** is the natural parameter for q(phi_m)
- `log_phi`: E_q[log(phi)]
    - *J x M* matrix stored as a 2D numpy array
    - The m-th **column** is E_q[log(phi_m)]
- `negative_kl_q_p_phi`: -KL(q(phi_m) || prior(phi_m))
    - *M*-element vector stored as a 1D numpy array

#### Related to q(alpha) = Normal_M
Note: Each q(alpha_i) is a `normal_v` distribution with *M* elements
- `mu_q_alpha`: E_q[alpha]
    - *B x M* matrix stored as a 2D numpy array
    - The b-th row is E_q[alpha_ib]
- `sigma_sq_q_alpha`: V_q[alpha]
    - *B x M* matrix stored as a 2D numpy array
    - The b-th row is V_q[alpha_ib]
- `alpha_sq`: E_q[alpha^2]
    - *B x M* matrix stored as a 2D numpy array
    - The b-th row is E_q[alpha_ib^2]
- `log_theta_denom_approx`: E_q[log(softmax(alpha))]
    - *B*-element vector stored as a 1D numpy array
    - The b-th element is E_q[log(softmax(alpha_ib))]
- `entropy_q_alpha`: Entropy(q(alpha_ib))
    - *B*-element vector stored as a 1D numpy array
    - The b-th element is the entropy of q(alpha_ib)

#### Related to q(tau_alpha) = Gamma_M
Note: q(tau_alpha) is a `gamma_v` distribution with *M* elements
- `eta_q_tau_alpha`: Natural variational parameter for q(tau_alpha)
    - *2M*-element vector stored as a 1D numpy array
- `log_tau_alpha`: E_q[log(tau_alpha)]
    - *M*-element vector stored as a 1D numpy array
- `tau_alpha`: E_q[tau_alpha]
    - *M*-element vector stored as a 1D numpy array
- `negative_kl_q_p_tau_alpha`: -KL(q(tau_alpha) || prior(tau_alpha))
    - Scalar stored as a 0D numpy array
    - Equal to -\sum_m KL(q(tau_alpha_m) || prior(tau_alpha_m))

#### Related to q(kappa) = MultivariateNormal_M
- `kappa`: E_q[kappa]
    - *I x M* matrix stored as a 2D numpy array
    - The i-th row is E_q[kappa_i]
- `kappa_sq`: E_q[kappa^2]
    - *I x M* matrix stored as a 2D numpy array
    - The i-th row is E_q[kappa_i^2]
- `kappa_outer`: E_q[outer(kappa_i, kappa_i)] for i=1...I
    - *I x M x M* matrix stored as a 3D numpy array
    - The i-th row is E_q[outer(kappa_i, kappa_i)]
- `entropy_q_kappa`: Entropy(q(kappa_i)) for i=1...I
    - *I*-element vector stored as a 1D numpy array

#### Related to q(mu_kappa) = MultivariateNormal_M
- `eta_q_mu_kappa`: Natural variational parameter for q(mu_kappa)
    - *M + M^2*-element vector stored as a 1D numpy array
- `mu_kappa`: E_q[mu_kappa]
    - *M*-element vector stored as a 1D numpy array
- `mu_kappa_outer`: E_q[outer(mu_kappa, mu_kappa)]
    - *M x M* matrix stored as a 2D numpy array
- `negative_kl_q_p_mu_kappa`: -KL(q(mu_kappa) || prior(mu_kappa))
    - Scalar stored as a 0D numpy array

#### Related to q(lambda_kappa) = Wishart_M
- `eta_q_lambda_kappa`: Natural variational parameter for q(lambda_kappa)
    - *1 + M^2*-element vector stored as a 1D numpy array
- `lambda_kappa`: E_q[lambda_kappa]
    - *M x M* matrix stored as a 2D numpy array
- `log_det_lambda_kappa`: E_q[log(det(lambda_kappa))]
    - Scalar stored as a 0D numpy array
- `negative_kl_q_p_lambda_kappa`: -KL(q(lambda_kappa) || prior(lambda_kappa))
    - Scalar stored as a 0D numpy array

#### Related to q(beta_m) = MultivariateNormal_|X|
- `eta_q_beta`: Natural variational parameter for each q(beta_m)
    - *M x (|X| + |X|^2)* matrix stored as a 2D numpy array
    - The m-th row is the natural parameter for q(beta_m)
- `beta`: E_q[beta]
    - *M x |X|* matrix stored as a 2D numpy array
    - The m-th row is E_q[beta_m]
- `beta_outer`: E_q[outer(beta_m, beta_m)] for m=1...M
    - *M x |X| x |X|* matrix stored as a 3D numpy array
    - The m-th row is E_q[outer(beta_m, beta_m)]
- `negative_kl_q_p_beta`: -KL(q(beta_m) || prior(beta_m))
    - *M*-element vector stored as a 1D numpy array

#### Related to q(gamma_m) = MultivariateNormal_|W|
- `eta_q_gamma`: Natural variational parameter for each q(gamma_m)
    - *M x (|W| + |W|^2)* matrix stored as a 2D numpy array
    - The m-th row is the natural parameter for q(gamma_m)
- `gamma`: E_q[gamma]
    - *M x |W|* matrix stored as a 2D numpy array
    - The m-th row is E_q[gamma_m]
- `gamma_outer`: E_q[outer(gamma_m, gamma_m)] for m=1...M
    - *M x |W| x |W|* matrix stored as a 3D numpy array
    - The m-th row is E_q[outer(gamma_m, gamma_m)]
- `negative_kl_q_p_gamma`: -KL(q(gamma_m) || prior(gamma_m))
    - *M*-element vector stored as a 1D numpy array

#### Related to q(rho_m) = MultivariateNormal_M
- `eta_q_rho`: Natural variational parameter for each q(rho_m)
    - *M x (M + M^2)* matrix stored as a 2D numpy array
    - The m-th row is the natural parameter for q(rho_m)
- `rho`: E_q[rho]
    - *M x M* matrix stored as a 2D numpy array
    - The m-th row is E_q[rho_m]
- `rho_outer`: E_q[outer(rho_m, rho_m)] for m=1...M
    - *M x M x M* matrix stored as a 3D numpy array
    - The m-th row is E_q[outer(rho_m, rho_m)]
- `negative_kl_q_p_rho`: -KL(q(rho_m) || prior(rho_m))
    - *M*-element vector stored as a 1D numpy array

#### Related to q(delta_m) = Normal_M
Note: q(delta_m) is a `normal_v` distribution with *M* elements
- `eta_q_delta`: Natural variational parameter for q(delta_m)
    - *2M*-element vector stored as a 1D numpy array
- `delta`: E_q[delta]
    - *M*-element vector stored as a 1D numpy array
- `delta_sq`: E_q[delta^2]
    - *M*-element vector stored as a 1D numpy array
- `negative_kl_q_p_delta`: -KL(q(delta) || prior(delta))
    - Scalar stored as a 0D numpy array

#### Related to q(delta_kappa) = Normal_1
Note: q(delta_kappa) is a `normal_v` distribution with *1* element
- `eta_q_delta_kappa`: Natural variational parameter for q(delta_kappa)
    - *2*-element vector stored as a 1D numpy array
- `delta_kappa`: E_q[delta_kappa]
    - Scalar stored as a 0D numpy array
- `delta_kappa_sq`: E_q[delta_kappa^2]
    - Scalar stored as a 0D numpy array
- `negative_kl_q_p_delta_kappa`: -KL(q(delta_kappa) || prior(delta_kappa))
    - Scalar stored as a 0D numpy array

#### Related to q(delta_beta) = Normal_1
Note: q(delta_beta) is a `normal_v` distribution with *1* element
- `eta_q_delta_beta`: Natural variational parameter for q(delta_beta)
    - *2*-element vector stored as a 1D numpy array
- `delta_beta`: E_q[delta_beta]
    - Scalar stored as a 0D numpy array
- `delta_beta_sq`: E_q[delta_beta^2]
    - Scalar stored as a 0D numpy array
- `negative_kl_q_p_delta_beta`: -KL(q(delta_beta) || prior(delta_beta))
    - Scalar stored as a 0D numpy array

#### Related to q(delta_gamma) = Normal_1
Note: q(delta_gamma) is a `normal_v` distribution with *1* element
- `eta_q_delta_gamma`: Natural variational parameter for q(delta_gamma)
    - *2*-element vector stored as a 1D numpy array
- `delta_gamma`: E_q[delta_gamma]
    - Scalar stored as a 0D numpy array
- `delta_gamma_sq`: E_q[delta_gamma^2]
    - Scalar stored as a 0D numpy array
- `negative_kl_q_p_delta_gamma`: -KL(q(delta_gamma) || prior(delta_gamma))
    - Scalar stored as a 0D numpy array

#### Step sizes for the gradient updates q(alpha) 
- `ss_mu_q_alpha`: Adaptive step sizes for the gradients of mu_q_alpha_ib 
    - *B*-element vector stored as a 1D numpy array
    - The b-th element gives the step size for the gradient of mu_q_alpha_ib
- `ss_log_sigma_q_alpha`: Adaptive step size for the gradients of log_sigma_q_alpha_ib 
    - *B*-element vector stored as a 1D numpy array
    - The b-th element gives the step size for the gradient of log_sigma_q_alpha_ib


### Initial state

The variational state as described above, at the start of the variational
inference algorithm, written to:
```
output/M$M/$MODEL/initial_state.npz
```


### Prior

The prior information used in the variational inference algorithm, based on the
settings as specified in **`model/prior.py`**, written to:
```
output/M$M/$MODEL/prior.npz
```


### Data

The dataset used in the variational inference algorithm, based on the input
data from **y.csv**, **x.csv**, and **w.csv**, written to:
```
output/M$M/$MODEL/data.npz
```


### Miscellaneous settings

The miscellaneous settings as specified in **`settings.py`**, written to:
```
output/M$M/$MODEL/misc_settings.npz
```


### VI settings

The variational inference settings as specified in **`settings.py`**, written
to:
```
output/M$M/$MODEL/vi_settings.npz
```
