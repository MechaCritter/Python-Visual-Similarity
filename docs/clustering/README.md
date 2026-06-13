# Clustering models

The `pyvisim.clustering` module holds the small models the encoders use to build
their visual vocabulary and to reduce dimensionality. Each one owns an underlying
scikit-learn estimator and exposes just the attributes the encoders need through
typed getters, so the encoders never touch scikit-learn's `*_` fitted attributes
directly.

| Object | File | Backed by | Used by |
|--------|------|-----------|---------|
| `KMeans` | [`kmeans.py`](../../pyvisim/clustering/kmeans.py) | `sklearn.cluster.KMeans` | `VLADEncoder` |
| `GaussianMixtureModel` | [`gmm.py`](../../pyvisim/clustering/gmm.py) | `sklearn.mixture.GaussianMixture` | `FisherVectorEncoder` |
| `PCA` | [`pca.py`](../../pyvisim/clustering/pca.py) | `sklearn.decomposition.PCA` | both encoders (optional) |

`KMeans` and `GaussianMixtureModel` are clustering models and share
[`ClusteringModelBase`](../../pyvisim/clustering/_base_clustering.py). `PCA` is not a
clustering model, so it sits directly on the shared `_SklearnModelBase` instead.

## How they work

You create a model unfitted, passing the scikit-learn constructor parameters straight
through:

```python
from pyvisim.clustering import KMeans, GaussianMixtureModel, PCA

kmeans = KMeans(n_clusters=256, random_state=0)
gmm = GaussianMixtureModel(n_components=256, random_state=0)
pca = PCA(n_components=64, whiten=True)
```

`n_clusters` / `n_components` are explicit; everything else is forwarded verbatim to
the wrapped estimator. Call `fit(features)` to train, and check `is_fitted` at any
time. The fitted-only getters (`cluster_centers`, `weights`, `means`, `covariances`,
`n_features_in`, ...) raise `NotFittedError` if you read them before fitting, so you
get a clear error instead of an `AttributeError` from scikit-learn.

In normal use you don't build these yourself: the encoders create the matching model
for you from the parameters passed to their constructors (see
[encoders/base_encoder.md](../encoders/base_encoder.md)).

## What each model exposes

**`KMeans`**
- `n_clusters` — number of clusters.
- `cluster_centers` — `(n_clusters, n_features)` centroid coordinates.
- `predict(features)` — nearest cluster index per row. This is the hard assignment
  VLAD uses.

**`GaussianMixtureModel`**
- `n_clusters` — number of mixture components (the `ClusteringModelBase` name for it).
- `weights`, `means`, `covariances` — the GMM parameters the Fisher Vector gradients
  are computed from.
- `predict_proba(features)` — posterior probability per component, i.e. the soft
  assignment Fisher Vectors use.
- Diagonal covariance only. Asking for any other `covariance_type` raises `ValueError`
  up front, because the Fisher Vector math assumes diagonal covariances (and training
  is much faster that way).

**`PCA`**
- `n_components` — number of components of the fitted PCA.
- `transform(features)` — projects features onto the principal components.

## Adopting a pretrained scikit-learn estimator

There's an internal `_from_sklearn` classmethod used to wrap an already-fitted
estimator loaded from a legacy `KMeansWeights` / `GMMWeights` pickle. It type-checks
the estimator (and, for the GMM, re-validates the diagonal covariance) before adopting
it. You won't call this directly; it backs the deprecated weight-loading path described
in [encoders/weights.md](../encoders/weights.md).
