# FisherVectorEncoder

File: [`fisher_vector.py`](../../pyvisim/encoders/fisher_vector.py)

The Fisher Vector encodes an image into a vector of shape `(2 * K * D + K,)`, where
`K` is the number of GMM components and `D` is the local descriptor dimension (after
optional PCA). The `2 * K * D` term comes from the mean and variance gradients, and
the `+ K` term from the mixture-weight gradients.

## Constructing one

Fisher Vectors always cluster with a Gaussian Mixture Model, configured through the
encoder:

```python
from pyvisim.encoders import FisherVectorEncoder

fisher = FisherVectorEncoder(
    n_components=256,                # number of mixture components
    gmm_params={"random_state": 0},  # forwarded to sklearn.mixture.GaussianMixture
    pca_params={"n_components": 64}, # optional; omit for no PCA
)
fisher.learn(images)                 # fits the PCA (if any) then the GMM
```

`n_components` is passed directly, not inside `gmm_params` (doing both raises a
`ValueError`). The GMM uses diagonal covariances; passing any other `covariance_type`
in `gmm_params` raises a `ValueError`, since the Fisher Vector math assumes diagonal
covariances. Save a fitted encoder with `fisher.save_to_disk("fisher")` and reload it
with `FisherVectorEncoder.load_from_disk("fisher.encoder")`, see
[base_encoder.md](base_encoder.md).

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
