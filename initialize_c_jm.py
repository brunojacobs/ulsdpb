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
import argparse
import logging
import os
import time

# External modules
import numpy as np

import lda
import lda.utils

# Own modules
import settings

assert lda.__version__ == '2.0.0', 'lda package should be 2.0.0'

def _fit_alt(
        lda_obj,
        X,
        output_folder,
        n_burn_in,
):
    """Adaptation of the _fit method for an LDA object.

    The change is that the MCMC draws are averaged and saved, instead of just
    storing the final MCMC draw. The first n_burn_in samples are skipped.

    The average draw is stored as a J x M CSV of floats.

    Parameters
        lda_obj: The LDA object.
        X: Data as expected by the _fit method of the LDA object.
        output_folder: Folder where the average draws are stored
        save_every_nth: Saves the average over the last n-draws.
    """

    random_state = lda.utils.check_random_state(lda_obj.random_state)
    rands = lda_obj._rands.copy()
    lda_obj._initialize(X)

    # ULSDPB ADDITION: START

    logger = logging.getLogger('lda')

    # nzw_: M x J topic-word assignments
    # ndz_: I x M document-topic assignments

    # Keep cache variables used to store output of multiple iterations
    sum_nzw_ = np.zeros_like(lda_obj.nzw_)
    sum_ndz_ = np.zeros_like(lda_obj.ndz_)

    # ULSDPB ADDITION: END

    for it in range(lda_obj.n_iter):
        # FIXME: using numpy.roll with a random shift might be faster
        random_state.shuffle(rands)
        if it % lda_obj.refresh == 0:
            ll = lda_obj.loglikelihood()
            logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
            # keep track of loglikelihoods for monitoring convergence
            lda_obj.loglikelihoods_.append(ll)
        lda_obj._sample_topics(rands)

        # ULSDPB ADDITION: START
        sum_nzw_ += lda_obj.nzw_
        sum_ndz_ += lda_obj.ndz_

        if (it + 1) == n_burn_in:
            # Reset the sum
            sum_nzw_[:] = 0
            sum_ndz_[:] = 0

        if (it + 1) == lda_obj.n_iter:

            assert np.isclose(
                np.sum(sum_nzw_),
                np.sum(X) * (lda_obj.n_iter - n_burn_in)
            )

            average_c_jm = sum_nzw_.T / (lda_obj.n_iter - n_burn_in)
            assert np.isclose(
                np.sum(average_c_jm),
                np.sum(X)
            )

            np.savetxt(
                os.path.join(output_folder, 'initial_c_jm.csv'),
                X=average_c_jm,
                fmt='%.18f',
                delimiter=','
            )

        # ULSDPB ADDITION: END

    ll = lda_obj.loglikelihood()
    logger.info("<{}> log likelihood: {:.0f}".format(lda_obj.n_iter - 1, ll))
    # note: numpy /= is integer division
    lda_obj.components_ = (lda_obj.nzw_ + lda_obj.eta).astype(float)
    lda_obj.components_ /= np.sum(lda_obj.components_, axis=1)[:, np.newaxis]
    lda_obj.topic_word_ = lda_obj.components_
    lda_obj.doc_topic_ = (lda_obj.ndz_ + lda_obj.alpha).astype(float)
    lda_obj.doc_topic_ /= np.sum(lda_obj.doc_topic_, axis=1)[:, np.newaxis]

    # delete attributes no longer needed after fitting to save memory and reduce clutter
    del lda_obj.WS
    del lda_obj.DS
    del lda_obj.ZS
    return lda_obj


parser = argparse.ArgumentParser()
parser.add_argument('-M', type=int)
parser.add_argument('-S', type=int)
parser.add_argument('-N_ITER', type=int)
parser.add_argument('-N_BURN', type=int)
parser_args = parser.parse_args()

M = parser_args.M
SEED = parser_args.S
N_ITER = parser_args.N_ITER
N_BURN = parser_args.N_BURN

assert N_ITER > N_BURN

print('Pseudocounts initialization based on Collapsed-Gibbs LDA')
print('Number of motivations:', M)
print('Seed:', SEED)
print('Total number of iterations:', N_ITER)
print('Burn-in iterations to be discarded:', N_BURN)

M_OUTPUT_FOLDER = os.path.join(settings.OUTPUT_FOLDER, 'M' + str(M))
if not os.path.exists(M_OUTPUT_FOLDER):
    os.makedirs(M_OUTPUT_FOLDER)

# Load purchase data
y = np.loadtxt(settings.Y_CSV, dtype=int, delimiter=',')

# Get dimensions
total_baskets = len(np.unique(y[:, 1]))
dim_j = len(np.unique(y[:, 2]))

# Convert to conventional purchase matrix
purchases = np.zeros((total_baskets, dim_j), dtype=np.int64)

ib = 0
for i, ib, y_ibn in y:
    purchases[ib, y_ibn] += 1

# Create LDA object
lda_object = lda.LDA(
    n_topics=M,
    n_iter=N_ITER,
    alpha=1/M,
    eta=1/dim_j,
    random_state=SEED,
    refresh=1000,
)

# Estimate the LDA model
t_start = time.time()

_fit_alt(
    lda_obj=lda_object,
    X=purchases,
    output_folder=M_OUTPUT_FOLDER,
    n_burn_in=N_BURN,
)

t_end = time.time()

print('Pseudocounts initialization took {} seconds.'.format(t_end - t_start))
