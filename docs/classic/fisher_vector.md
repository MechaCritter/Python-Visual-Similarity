# FisherVectorEmbedder

The Fisher Vector solves both problems of the BoW model. First, it encodes higher-order statistics, such as the first and optionally second-order differences, instead of just counting the occurrences of visual words like BoW. This method is derived from the Fisher kernel framework, which describes a sample set's deviation from an average distribution. Secondly, the distribution of the local descriptors, unlike BoW and VLAD, is modeled by a Gaussian Mixture Model. This mitigates the hard assignment problem introduced by the K-Means algorithm, since each descriptor is assigned to multiple Gaussian components with different probabilities.

## Computation

### Fisher Kernel Framework

Given a set of $T$ local descriptors $X = \{x_t; t = 1, \ldots, T\}$ extracted from an image, it is assumed that the generation process of $X$ can be modeled by an image-independent probability density function $u_{\lambda}$ with parameters $\lambda$ [Jégou et al., 2012]. The gradient vector $G^{X}_{\lambda}$ is obtained by computing the gradient of the log-likelihood of the sample set $X$ with respect to the parameters $\lambda$:

$$
G^{X}_{\lambda} = \frac{1}{T} \nabla_{\lambda} \log u_{\lambda}(X)
$$

where $G^{X}_{\lambda}$ describes the contribution of the parameters to the generation process [Perronnin & Dance, 2010].

The Fisher kernel is then defined as:

$$
K(X, Y) = (G^{X}_{\lambda})^T F_{\lambda}^{-1} G^{Y}_{\lambda}
$$

where $F_{\lambda}$ is the Fisher information matrix, defined by:

$$
F_{\lambda} = \mathbb{E}_{x \sim u_{\lambda}} \left[ \nabla_{\lambda} \log u_{\lambda}(x) \nabla_{\lambda} \log u_{\lambda}(x)^T \right]
$$

$\mathcal{G}^{X}_{\lambda}$ is the Fisher Vector after applying the Cholesky decomposition on $F_{\lambda}^{-1} = L_{\lambda}^T L_{\lambda}$, and is computed as:

$$
\mathcal{G}_i^X = L_{\lambda} G^{X}_{\lambda}
$$

### Fisher Vector Computation

As discussed, the Fisher Vector encodes each descriptor to multiple Gaussian components (also called "soft assignment"). The probability of a descriptor $x_t$ belonging to the $i$-th Gaussian is computed with the Gaussian Mixture Model.

The Gaussian Mixture Model is chosen for $u_{\lambda}(x) = \sum_{i=1}^{K} w_i u_i(x)$, where $w_i, \mu_i, \Sigma_i$ are the mixture weights, mean vectors, and variance matrices of the Gaussian $u_i$. The Fisher Vector is then computed as:

$$
\gamma_t(i) = \frac{w_i u_i(x_t)}{\sum_{j=1}^{K} w_j u_j(x_t)}
$$

$$
\mathcal{G}_i^X = \frac{1}{T \sqrt{w_i}} \sum_{t=1}^{T} \gamma_t(i)\, \sigma_i^{-1} (x_t - \mu_i)
$$

where:

- $\gamma_t(i)$ is the soft assignment of descriptor $x_t$ to the $i$-th Gaussian.
- $w_i$, $\mu_i$, and $\Sigma_i$ are the mixture weight, mean vector, and covariance matrix of the $i$-th Gaussian component.

The final Fisher Vector $G^{X}_{\lambda}$ is the concatenation of the vectors $G^{X}_{i}$ for $i = 1, \ldots, K$, resulting in a $K \times d$-dimensional vector. This vector captures both the occurrence and distributional properties of the local descriptors.

The resulting vector has shape `(2 * K * D + K,)`, where
`K` is the number of GMM components and `D` is the local descriptor dimension (after
optional PCA).

## Usage

```python
from pyvisim.classic import FisherVectorEmbedder

fisher = FisherVectorEmbedder(
    n_components=256,                # number of mixture components
    gmm_params={"rng": 0},           # forwarded to the GMM
    pca_params={"n_components": 64}, # optional; omit for no PCA
)
fisher.learn(images)                 # fits the PCA (if any) then the GMM

embedding = fisher.embed(image)      # Embed image into a Fisher Vector

similarity = fisher.similarity_score(image1, image2)  # Cosine similarity between two images

fisher.save_to_disk("fisher.embedder")    # Save the embedder to disk

fisher = FisherVectorEmbedder.load_from_disk("fisher.embedder")  # Load the embedder from disk
```

## GMM parameters (`gmm_params`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_init` | `1` | Number of k-means++ seeded EM runs; the run with the highest final log-likelihood is kept. Raise it for better, more stable vocabularies. |
| `max_iter` | `100` | Maximum number of EM iterations per run. |
| `tol` | `1e-3` | Convergence threshold: a run stops when the change of the mean per-sample log-likelihood between iterations falls below it. |
| `reg_covar` | `1e-6` | Non-negative regularisation added to (and floored on) the per-feature variances, keeping them strictly positive when a component collapses or dies. |
| `rng` | `None` | Seed (`int`) or `numpy.random.Generator` for reproducible fitting. |

## PCA parameters (`pca_params`)

See [PCA](https://mechacritter.github.io/Python-Visual-Similarity/classic/pca.html).

## References

- H. Jégou et al. "Aggregating Local Image Descriptors into Compact Codes". In: IEEE
  Transactions on Pattern Analysis and Machine Intelligence 34.9 (2012), pp. 1704-1716.
  doi: 10.1109/TPAMI.2011.235.
