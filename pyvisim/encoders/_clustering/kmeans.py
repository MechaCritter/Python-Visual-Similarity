"""K-Means clustering class used by the VLAD encoder."""

import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from scipy.cluster.vq import kmeans, vq

from ...typing import FloatNumpyArray, IntNumpyArray
from ._base_clustering import ClusteringModelBase, _decode, _encode

_KMeansT = TypeVar("_KMeansT", bound="KMeans")

_INIT_PARAM_NAMES = ("n_init", "thresh", "check_finite", "rng")


def _n_init_from_sklearn(n_init: Any) -> int:
    """Translates a scikit-learn ``n_init`` value into an integer seeding count."""
    if n_init == "auto":
        return 1
    if isinstance(n_init, (int, np.integer)):
        return int(n_init)
    raise TypeError(
        f"Unsupported scikit-learn n_init {n_init!r}; only an integer or "
        "'auto' can be mapped to a seeding count."
    )


def _rng_from_sklearn(random_state: Any) -> int | None:
    """Translates a scikit-learn ``random_state`` into a NumPy ``rng`` seed."""
    if random_state is None:
        return None
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    raise TypeError(
        f"Unsupported scikit-learn random_state {random_state!r}; only an "
        "integer seed or None can be mapped to rng."
    )


_SKLEARN_PARAM_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "n_clusters": ("n_clusters", int),
    "tol": ("thresh", float),
    "n_init": ("n_init", _n_init_from_sklearn),
    "random_state": ("rng", _rng_from_sklearn),
}


def _kmeans_plusplus(
    data: FloatNumpyArray, n_clusters: int, generator: np.random.Generator
) -> FloatNumpyArray:
    """
    Selects initial centroids with greedy k-means++ seeding.

    :param data: Observation matrix of shape (n_samples, n_features).
    :param n_clusters: Number of centroids to select.
    :param generator: NumPy random generator driving the candidate draws.
    :return: Initial centroids of shape (n_clusters, n_features).
    """
    n_samples = data.shape[0]
    n_local_trials = 2 + int(np.log(n_clusters))
    centers = np.empty((n_clusters, data.shape[1]), dtype=data.dtype)
    # ||x||^2 per sample, reused by the expanded squared-distance formula
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 (avoids broadcasting an
    # (n_trials, n_samples, n_features) difference tensor per center).
    squared_norms = np.einsum("ij,ij->i", data, data)

    centers[0] = data[generator.integers(n_samples)]
    closest_dist_sq = ((data - centers[0]) ** 2).sum(axis=1)
    current_potential = closest_dist_sq.sum()

    for center_index in range(1, n_clusters):
        if current_potential == 0.0:
            # All remaining points coincide with chosen centroids.
            candidate_ids = generator.integers(n_samples, size=n_local_trials)
        else:
            draws = generator.random(n_local_trials) * current_potential
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), draws)
            np.clip(candidate_ids, None, n_samples - 1, out=candidate_ids)

        # Potential of each candidate: (n_local_trials, n_samples) distances,
        # clipped against the current closest-centroid distances.
        dist_sq = (
            squared_norms[candidate_ids, None]
            - 2.0 * data[candidate_ids] @ data.T
            + squared_norms[None, :]
        )
        np.clip(dist_sq, 0.0, None, out=dist_sq)  # cancellation can dip below 0
        np.minimum(dist_sq, closest_dist_sq, out=dist_sq)
        candidate_potentials = dist_sq.sum(axis=1)

        best = int(np.argmin(candidate_potentials))
        current_potential = candidate_potentials[best]
        closest_dist_sq = dist_sq[best]
        centers[center_index] = data[candidate_ids[best]]
    return centers


class KMeans(ClusteringModelBase):
    """
    K-Means clustering model, used by the VLAD encoder.

    :param n_clusters: Number of clusters to form.
    :param n_init: Number of k-means++ seedings to run; each seeded codebook
        is refined and the result with the lowest distortion is kept.
    :param thresh: Terminates each refinement when the change in distortion
        falls below this threshold.
    :param check_finite: Whether to check that the input contains only
        finite numbers. Disabling may give a performance gain.
    :param rng: Seed (int) or :class:`numpy.random.Generator` for
        reproducible fitting.
    :raises ValueError: If ``n_clusters`` or ``n_init`` is not positive.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        *,
        n_init: int = 1,
        thresh: float = 1e-05,
        check_finite: bool = True,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}.")
        if n_init < 1:
            raise ValueError(f"n_init must be positive, got {n_init}.")
        self._n_clusters = int(n_clusters)
        self._n_init = int(n_init)
        self._thresh = thresh
        self._check_finite = check_finite
        self._rng = rng
        self._cluster_centers: FloatNumpyArray | None = None
        self._scale: FloatNumpyArray | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted (centroids are stored)."""
        return self._cluster_centers is not None

    @property
    def n_clusters(self) -> int:
        """Number of clusters of the K-Means model."""
        return self._n_clusters

    @property
    def n_features_in(self) -> int:
        """Number of features the fitted model expects as input."""
        return int(self.cluster_centers.shape[1])

    @property
    def cluster_centers(self) -> FloatNumpyArray:
        """Cluster centroids, shape (n_clusters, n_features)."""
        self._check_is_fitted()
        return cast(FloatNumpyArray, self._cluster_centers)

    @property
    def _fitted_scale(self) -> FloatNumpyArray:
        """Per-feature whitening scale from training, shape (n_features,)."""
        self._check_is_fitted()
        return cast(FloatNumpyArray, self._scale)

    @staticmethod
    def _whitening_scale(data: FloatNumpyArray) -> FloatNumpyArray:
        """
        Computes the per-feature whitening scale of the training data.

        :param data: Feature matrix of shape (n_samples, n_features).
        :return: Per-feature standard deviations, shape (n_features,).
        """
        scale = np.asarray(data.std(axis=0))
        scale[scale == 0.0] = 1.0
        return scale

    def fit(self, features: FloatNumpyArray) -> None:
        """
        Learns the cluster centroids from the given feature matrix.

        The features are whitened, then ``n_init`` k-means++ seedings
        are each refined and the codebook
        with the lowest distortion is kept. The centroids are stored
        de-whitened, i.e. in the original feature space, together with the
        whitening scale used at prediction time.

        :param features: Feature matrix of shape (n_samples, n_features).
        :raises ValueError: If there are fewer samples than clusters.
        """
        data = np.asarray(features)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)
        if data.shape[0] < self._n_clusters:
            raise ValueError(
                f"Cannot form {self._n_clusters} clusters from {data.shape[0]} samples."
            )

        scale = self._whitening_scale(data)
        whitened = data / scale
        generator = np.random.default_rng(self._rng)

        best_codebook: FloatNumpyArray | None = None
        best_distortion = np.inf
        for _ in range(self._n_init):
            guess = _kmeans_plusplus(whitened, self._n_clusters, generator)
            codebook, distortion = kmeans(
                whitened,
                guess,
                thresh=self._thresh,
                check_finite=self._check_finite,
            )
            if best_codebook is None or distortion < best_distortion:
                best_codebook, best_distortion = codebook, distortion
        best_codebook = cast(FloatNumpyArray, best_codebook)

        if best_codebook.shape[0] < self._n_clusters:
            warnings.warn(
                f"SciPy's k-means dropped {self._n_clusters - best_codebook.shape[0]} "
                f"empty cluster(s); the model now has {best_codebook.shape[0]} "
                "clusters.",
                FutureWarning,
                stacklevel=2,
            )
            self._n_clusters = int(best_codebook.shape[0])
        self._cluster_centers = best_codebook * scale
        self._scale = scale

    def predict(self, features: FloatNumpyArray) -> IntNumpyArray:
        """
        Predicts the closest cluster for each feature vector.

        The features are whitened with the scale computed during fitting —
        observations must be whitened the same way as the codebook — and
        then assigned to the nearest centroid.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Cluster index of each sample, shape (n_samples,).
        :raises NotFittedError: If the model is not fitted.
        """
        centers = self.cluster_centers
        scale = self._fitted_scale
        data = np.asarray(features, dtype=centers.dtype)
        labels, _ = vq(data / scale, centers / scale, check_finite=self._check_finite)
        return np.asarray(labels)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the fitted model into a JSON-safe dictionary.

        The state layout keeps the ``cluster_centers_`` / ``n_features_in_``
        key names used by older releases, so the ``.encoder`` files they
        wrote can still be read; the whitening scale is stored under
        ``scale_``.

        **NOTE**: ``rng`` is stored only when it is an integer seed; a
        :class:`numpy.random.Generator` is stateful and not JSON-safe.

        :return: A dictionary describing the model class and its fitted state.
        :raises NotFittedError: If the model is not fitted.
        """
        state = {
            "cluster_centers_": self.cluster_centers,
            "scale_": self._fitted_scale,
            "n_clusters": self._n_clusters,
            "n_features_in_": self.n_features_in,
            "n_init": self._n_init,
            "thresh": self._thresh,
            "check_finite": self._check_finite,
            "rng": self._rng if isinstance(self._rng, int) else None,
        }
        return {
            "__class__": type(self).__name__,
            "__module__": type(self).__module__,
            "state": _encode(state),
        }

    @classmethod
    def from_dict(cls: type[_KMeansT], data: dict[str, Any]) -> _KMeansT:
        """
        Rebuilds a model from a dictionary produced by :meth:`to_dict`.

        Dictionaries serialised by older releases are also accepted: the
        centroids are restored, the whitening scale defaults to ones (those
        models were trained without whitening) and the remaining parameters
        keep their defaults.

        :param data: A mapping with the form: {"__class__": str, "__module__": str, "state": dict}.
        :return: A fitted model.
        :raises TypeError: If ``data`` is not a dictionary.
        :raises ValueError: If ``data`` is malformed or describes a different
            model type than this class expects.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dict from to_dict(), got {type(data).__name__}."
            )
        for key in ("__class__", "state"):
            if key not in data:
                raise ValueError(f"Malformed model dict; missing key {key!r}.")
        if data["__class__"] != cls.__name__:
            raise ValueError(
                f"{cls.__name__} expects a serialised {cls.__name__!r}, "
                f"got {data['__class__']!r}."
            )
        state = _decode(data["state"])
        if "cluster_centers_" not in state:
            raise ValueError("Malformed KMeans state; missing 'cluster_centers_'.")
        centers = np.asarray(state["cluster_centers_"])
        params = {name: state[name] for name in _INIT_PARAM_NAMES if name in state}
        if "n_init" in params:
            # Legacy scikit-learn states share the "n_init" key but may hold
            # the sentinel "auto" instead of an integer.
            params["n_init"] = _n_init_from_sklearn(params["n_init"])
        model = cls(n_clusters=int(state.get("n_clusters", centers.shape[0])), **params)
        model._cluster_centers = centers
        model._scale = np.asarray(
            state.get("scale_", np.ones(centers.shape[1], dtype=centers.dtype))
        )
        return model

    @classmethod
    def from_sklearn(cls: type[_KMeansT], model: Any) -> _KMeansT:
        """
        Creates a model from a :class:`sklearn.cluster.KMeans` estimator.

        The scikit-learn constructor arguments are translated to their
        equivalents here (``tol`` becomes ``thresh``, ``n_init`` the number
        of k-means++ seedings — ``"auto"`` resolves to 1 — and
        ``random_state`` becomes ``rng``). ``max_iter`` has no counterpart —
        each refinement iterates until the distortion change falls below
        ``thresh`` — and is ignored, as are ``algorithm``, ``copy_x`` and
        ``verbose``. Only the default
        ``init="k-means++"`` is accepted, since k-means++ is the only
        initialization supported.

        If the estimator is fitted, its ``cluster_centers_`` are adopted
        with a whitening scale of ones — scikit-learn trains on unwhitened
        data — so the returned model predicts identically to the estimator.

        :param model: A :class:`sklearn.cluster.KMeans` instance, fitted or not.
        :return: A model configured (and, if ``model`` is fitted, fitted) from
            the given estimator.
        :raises TypeError: If ``model`` is not a :class:`sklearn.cluster.KMeans`
            instance, or its ``random_state`` / ``n_init`` cannot be translated.
        :raises ValueError: If its ``init`` is anything but ``"k-means++"``.
        """
        # Imported lazily: this module is otherwise scikit-learn-free, and the
        # conversion path is the only one that needs the estimator class.
        from sklearn.cluster import KMeans as _SklearnKMeans

        if not isinstance(model, _SklearnKMeans):
            raise TypeError(
                f"{cls.__name__}.from_sklearn expects a sklearn.cluster.KMeans "
                f"instance, not {type(model).__name__}."
            )
        if not (isinstance(model.init, str) and model.init == "k-means++"):
            raise ValueError(
                f"Unsupported scikit-learn init {model.init!r}; k-means++ is "
                "the only supported initialization."
            )
        params = {
            name: convert(getattr(model, sklearn_name))
            for sklearn_name, (name, convert) in _SKLEARN_PARAM_MAP.items()
        }
        instance = cls(params.pop("n_clusters"), **params)
        centers = getattr(model, "cluster_centers_", None)
        if centers is not None:
            centers = np.asarray(centers)
            instance._cluster_centers = centers
            instance._scale = np.ones(centers.shape[1], dtype=centers.dtype)
        return instance

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_clusters={self._n_clusters}, "
            f"n_init={self._n_init}, fitted={self.is_fitted})"
        )
