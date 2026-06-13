# FisherVectorEncoder

File: [`fisher_vector.py`](../../pyvisim/encoders/fisher_vector.py)

The Fisher Vector encodes an image into a vector of shape `(2 * K * D + K,)`, where
`K` is the number of GMM components and `D` is the local descriptor dimension (after
optional PCA). The `2 * K * D` term comes from the mean and variance gradients, and
the `+ K` term from the mixture-weight gradients.

## How `encode` works

For each image:

1. Extract local descriptors (default `RootSIFT`) and apply PCA if set.
2. Compute **soft assignments**: the GMM posterior probability of each descriptor
   belonging to each component (`predict_proba`).
3. Accumulate the sufficient statistics (`pp_sum`, `pp_x`, `pp_x_2`) needed for the
   gradients.
4. Compute the gradients of the GMM log-likelihood with respect to its parameters:
   - `d_pi`: gradient w.r.t. mixture weights (first-order).
   - `d_mu`: gradient w.r.t. means (first-order).
   - `d_sigma`: gradient w.r.t. variances (second-order). This is what VLAD lacks.
5. Apply the analytical diagonal Fisher information normalization (dividing by the
   square roots involving the mixture weights and covariances).
6. Concatenate `[d_pi, d_mu, d_sigma]`, then power-normalize and L2-normalize.

## References

- H. Jégou et al. "Aggregating Local Image Descriptors into Compact Codes". In: IEEE
  Transactions on Pattern Analysis and Machine Intelligence 34.9 (2012), pp. 1704-1716.
  doi: 10.1109/TPAMI.2011.235.
