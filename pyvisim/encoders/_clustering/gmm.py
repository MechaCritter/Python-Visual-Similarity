"""
Diagonal-covariance Gaussian Mixture Model used by the Fisher Vector
encoder.
"""

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
from scipy.special import logsumexp

from ...typing import Float64NumpyArray, FloatNumpyArray, IntNumpyArray
from ._base_clustering import ClusteringModelBase, _decode, _encode
from .kmeans import _kmeans_plusplus, _rng_from_sklearn

_GaussianMixtureT = TypeVar("_GaussianMixtureT", bound="DiagCovarGaussianMixture")

#: Constructor arguments (besides ``n_components``) restored on deserialisation.
_INIT_PARAM_NAMES = ("n_init", "max_iter", "tol", "reg_covar", "rng")

#: Maps :class:`sklearn.mixture.GaussianMixture` constructor arguments to this
#: class' constructor arguments, with a converter per argument. scikit-learn
#: arguments without a counterpart here (``warm_start``, ``verbose``,
#: ``verbose_interval``) are ignored. ``covariance_type``, ``init_params`` and
#: the ``*_init`` arrays are validated separately.
_SKLEARN_PARAM_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "n_components": ("n_components", int),
    "tol": ("tol", float),
    "reg_covar": ("reg_covar", float),
    "max_iter": ("max_iter", int),
    "n_init": ("n_init", int),
    "random_state": ("rng", _rng_from_sklearn),
}


class _EMRun(NamedTuple):
    """Fitted parameters and diagnostics of a single EM run."""

    weights: Float64NumpyArray
    means: Float64NumpyArray
    covariances: Float64NumpyArray
    lower_bounds: list[float]
    converged: bool

    @property
    def lower_bound(self) -> float:
        """Final training lower bound of the run."""
        return self.lower_bounds[-1]


class DiagCovarGaussianMixture(ClusteringModelBase):
    """
    Gaussian Mixture clustering model with diagonal covariances, used by
    the Fisher Vector encoder.


    :param n_components: Number of mixture components.
    :param covariance_type: Present for scikit-learn compatibility; only
        ``"diag"`` is accepted.
    :param n_init: Number of k-means++ seeded EM runs; the run with the
        highest final log-likelihood is kept.
    :param max_iter: Maximum number of EM iterations per run.
    :param tol: Convergence threshold: a run stops when the change of the
        mean per-sample log-likelihood between iterations falls below it.
    :param reg_covar: Non-negative regularisation added to (and floored
        on) the variances, keeping them strictly positive.
    :param rng: Seed (int) or :class:`numpy.random.Generator` for
        reproducible fitting.
    :raises ValueError: If ``n_components``, ``n_init`` or ``max_iter``
        is not positive, ``tol`` or ``reg_covar`` is negative, or a
        ``covariance_type`` other than ``"diag"`` is requested.
    """

    def __init__(
        self,
        n_components: int = 256,
        *,
        covariance_type: str = "diag",
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        if covariance_type != "diag":
            raise ValueError(
                f"{type(self).__name__} only supports covariance_type='diag', "
                f"got {covariance_type!r}."
            )
        if n_components < 1:
            raise ValueError(f"n_components must be positive, got {n_components}.")
        if n_init < 1:
            raise ValueError(f"n_init must be positive, got {n_init}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        if reg_covar < 0:
            raise ValueError(f"reg_covar must be non-negative, got {reg_covar}.")
        self._n_components = int(n_components)
        self._n_init = int(n_init)
        self._max_iter = int(max_iter)
        self._tol = float(tol)
        self._reg_covar = float(reg_covar)
        self._rng = rng
        self._weights: Float64NumpyArray | None = None
        self._means: Float64NumpyArray | None = None
        self._covariances: Float64NumpyArray | None = None
        self._converged = False
        self._fit_lower_bounds: list[float] = []

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted (mixture parameters are stored)."""
        return self._means is not None

    @property
    def n_clusters(self) -> int:
        """Number of mixture components of the GMM."""
        return self._n_components

    @property
    def n_features_in(self) -> int:
        """Number of features the fitted model expects as input."""
        return int(self.means.shape[1])

    @property
    def weights(self) -> Float64NumpyArray:
        """Mixture weights of each component, shape (n_components,)."""
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._weights)

    @property
    def means(self) -> Float64NumpyArray:
        """Mean of each mixture component, shape (n_components, n_features)."""
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._means)

    @property
    def covariances(self) -> Float64NumpyArray:
        """
        Diagonal covariance of each component, shape (n_components, n_features).
        """
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._covariances)

    def _validate_features(
        self, features: FloatNumpyArray, *, n_features: int | None = None
    ) -> Float64NumpyArray:
        """Coerces a feature matrix to ``float64`` and validates its shape."""
        data = np.asarray(features, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got a {data.ndim}D array.")
        if data.shape[0] == 0:
            raise ValueError("Expected a non-empty feature matrix, got 0 samples.")
        if n_features is not None and data.shape[1] != n_features:
            raise ValueError(
                f"Expected {n_features} features per sample, got {data.shape[1]}."
            )
        return data

    @staticmethod
    def _weighted_log_prob(
        data: Float64NumpyArray,
        data_sq: Float64NumpyArray,
        weights: Float64NumpyArray,
        means: Float64NumpyArray,
        covariances: Float64NumpyArray,
    ) -> Float64NumpyArray:
        """Evaluates ``log(weight_k) + log N(x | mean_k, diag(covariance_k))``."""
        precisions = 1.0 / covariances
        log_det = np.sum(np.log(covariances), axis=1)
        mahalanobis = (
            np.sum(means**2 * precisions, axis=1)
            - 2.0 * data @ (means * precisions).T
            + data_sq @ precisions.T
        )
        n_features = data.shape[1]
        log_prob = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)
        return np.asarray(log_prob + np.log(weights))

    def _e_step(
        self,
        data: Float64NumpyArray,
        data_sq: Float64NumpyArray,
        weights: Float64NumpyArray,
        means: Float64NumpyArray,
        covariances: Float64NumpyArray,
    ) -> tuple[float, Float64NumpyArray]:
        """
        Computes the log-responsibilities and the mean log-likelihood.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param data_sq: Precomputed elementwise square of ``data``.
        :param weights: Mixture weights, shape (n_components,).
        :param means: Component means, shape (n_components, n_features).
        :param covariances: Diagonal covariances, shape (n_components, n_features).
        :return: The mean per-sample log-likelihood and the
            log-responsibilities of shape (n_samples, n_components).
        """
        weighted_log_prob = self._weighted_log_prob(
            data, data_sq, weights, means, covariances
        )
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        return float(np.mean(log_prob_norm)), np.asarray(log_resp)

    def _m_step(
        self,
        data: Float64NumpyArray,
        data_sq: Float64NumpyArray,
        resp: Float64NumpyArray,
    ) -> tuple[Float64NumpyArray, Float64NumpyArray, Float64NumpyArray]:
        """
        Re-estimates the mixture parameters from the responsibilities.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param data_sq: Precomputed elementwise square of ``data``.
        :param resp: Responsibilities, shape (n_samples, n_components).
        :return: Updated ``(weights, means, covariances)``.
        """
        nk = resp.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
        weights = nk / data.shape[0]
        means = resp.T @ data / nk[:, None]
        avg_sq = resp.T @ data_sq / nk[:, None]
        covariances = avg_sq - means**2 + self._reg_covar
        floor = self._reg_covar if self._reg_covar > 0.0 else np.finfo(np.float64).tiny
        np.maximum(covariances, floor, out=covariances)
        return weights, means, covariances

    def _initial_responsibilities(
        self, data: Float64NumpyArray, generator: np.random.Generator
    ) -> Float64NumpyArray:
        """
        Builds one-hot responsibilities from a greedy k-means++ seeding.

        Every sample is hard-assigned to its nearest seeded center, so the
        first M-step estimates each component's parameters from an actual
        cluster of samples.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param generator: NumPy random generator driving the seeding.
        :return: One-hot responsibilities, shape (n_samples, n_components).
        """
        centers = _kmeans_plusplus(data, self._n_components, generator)
        # argmin of ||x - c||^2 = argmin of ||c||^2 - 2 x.c (||x||^2 is constant
        # per sample), avoiding the (n_samples, k, D) broadcast.
        distances = np.sum(centers**2, axis=1) - 2.0 * data @ centers.T
        labels = np.argmin(distances, axis=1)
        resp = np.zeros((data.shape[0], self._n_components), dtype=np.float64)
        resp[np.arange(data.shape[0]), labels] = 1.0
        return resp

    def _run_em(
        self, data: Float64NumpyArray, generator: np.random.Generator
    ) -> _EMRun:
        """
        Runs one k-means++ seeded EM refinement to convergence.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param generator: NumPy random generator driving the seeding.
        :return: The fitted parameters and per-iteration lower bounds.
        """
        data_sq = np.square(data)
        resp = self._initial_responsibilities(data, generator)
        weights, means, covariances = self._m_step(data, data_sq, resp)

        lower_bounds: list[float] = []
        lower_bound = -np.inf
        converged = False
        for _ in range(self._max_iter):
            previous_lower_bound = lower_bound
            lower_bound, log_resp = self._e_step(
                data, data_sq, weights, means, covariances
            )
            weights, means, covariances = self._m_step(data, data_sq, np.exp(log_resp))
            lower_bounds.append(lower_bound)
            if abs(lower_bound - previous_lower_bound) < self._tol:
                converged = True
                break
        return _EMRun(weights, means, covariances, lower_bounds, converged)

    def fit(self, features: FloatNumpyArray) -> None:
        """
        Learns the mixture parameters from the given feature matrix.

        ``n_init`` EM runs are seeded with greedy k-means++ and the run
        with the highest final mean log-likelihood is kept.

        :param features: Feature matrix of shape (n_samples, n_features).
        :raises ValueError: If there are fewer samples than components or
            the matrix is not 2-dimensional.
        """
        data = self._validate_features(features)
        if data.shape[0] < self._n_components:
            raise ValueError(
                f"Cannot fit {self._n_components} mixture components on "
                f"{data.shape[0]} samples."
            )
        generator = np.random.default_rng(self._rng)

        best_run: _EMRun | None = None
        for _ in range(self._n_init):
            run = self._run_em(data, generator)
            if best_run is None or run.lower_bound > best_run.lower_bound:
                best_run = run
        best_run = cast(_EMRun, best_run)

        if not best_run.converged:
            warnings.warn(
                f"Best of {self._n_init} EM run(s) did not converge within "
                f"{self._max_iter} iterations; consider raising max_iter, "
                "raising tol or increasing n_init.",
                FutureWarning,
                stacklevel=2,
            )
        self._weights = best_run.weights
        self._means = best_run.means
        self._covariances = best_run.covariances
        self._converged = best_run.converged
        self._fit_lower_bounds = best_run.lower_bounds

    def _fitted_weighted_log_prob(self, features: FloatNumpyArray) -> Float64NumpyArray:
        """
        Evaluates the weighted log-probabilities under the fitted mixture.

        :raises NotFittedError: If the model is not fitted.
        :raises ValueError: If the feature count does not match the
            fitted model.
        """
        data = self._validate_features(features, n_features=self.n_features_in)
        return self._weighted_log_prob(
            data, np.square(data), self.weights, self.means, self.covariances
        )

    def predict_proba(self, features: FloatNumpyArray) -> Float64NumpyArray:
        """
        Evaluates the components' posterior probability for each feature vector.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Posterior probabilities, shape (n_samples, n_components).
        :raises NotFittedError: If the model is not fitted.
        :raises ValueError: If the feature count does not match the
            fitted model.
        """
        weighted_log_prob = self._fitted_weighted_log_prob(features)
        log_resp = weighted_log_prob - logsumexp(weighted_log_prob, axis=1)[:, None]
        return np.asarray(np.exp(log_resp))

    def predict(self, features: FloatNumpyArray) -> IntNumpyArray:
        """
        Predicts the most probable component for each feature vector.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Component index of each sample, shape (n_samples,).
        :raises NotFittedError: If the model is not fitted.
        :raises ValueError: If the feature count does not match the
            fitted model.
        """
        return np.asarray(self._fitted_weighted_log_prob(features).argmax(axis=1))

    def score(self, features: FloatNumpyArray) -> float:
        """
        Computes the mean per-sample log-likelihood under the fitted mixture.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Mean log-likelihood of the samples.
        :raises NotFittedError: If the model is not fitted.
        :raises ValueError: If the feature count does not match the
            fitted model.
        """
        weighted_log_prob = self._fitted_weighted_log_prob(features)
        return float(np.mean(logsumexp(weighted_log_prob, axis=1)))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the fitted model into a JSON-safe dictionary.

        **NOTE**: ``rng`` is stored only when it is an integer seed; a
        :class:`numpy.random.Generator` is stateful and not JSON-safe.

        :return: A dictionary describing the model class and its fitted state.
        :raises NotFittedError: If the model is not fitted.
        """
        state = {
            "weights_": self.weights,
            "means_": self.means,
            "covariances_": self.covariances,
            "n_components": self._n_components,
            "n_features_in_": self.n_features_in,
            "n_init": self._n_init,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "reg_covar": self._reg_covar,
            "rng": self._rng if isinstance(self._rng, int) else None,
        }
        return {
            "__class__": type(self).__name__,
            "__module__": type(self).__module__,
            "state": _encode(state),
        }

    @classmethod
    def from_dict(
        cls: type[_GaussianMixtureT], data: dict[str, Any]
    ) -> _GaussianMixtureT:
        """
        Rebuilds a model from a dictionary produced by :meth:`to_dict`.


        :param data: A mapping with the form: {"__class__": str, "__module__": str, "state": dict}.
        :return: A fitted model.
        :raises TypeError: If ``data`` is not a dictionary.
        :raises ValueError: If ``data`` is malformed, describes a different
            model type than this class expects, or holds a non-diagonal
            covariance type.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dict from to_dict(), got {type(data).__name__}."
            )
        for key in ("__class__", "state"):
            if key not in data:
                raise ValueError(f"Malformed model dict; missing key {key!r}.")
        if data["__class__"] not in (cls.__name__, "GaussianMixture"):
            raise ValueError(
                f"{cls.__name__} expects a serialised {cls.__name__!r} (or a "
                f"legacy 'GaussianMixture'), got {data['__class__']!r}."
            )
        state = _decode(data["state"])
        if state.get("covariance_type", "diag") != "diag":
            raise ValueError(
                f"{cls.__name__} only supports covariance_type='diag', "
                f"got {state['covariance_type']!r}."
            )
        for key in ("weights_", "means_", "covariances_"):
            if key not in state:
                raise ValueError(f"Malformed {cls.__name__} state; missing {key!r}.")
        means = np.asarray(state["means_"], dtype=np.float64)
        params = {name: state[name] for name in _INIT_PARAM_NAMES if name in state}
        if "rng" not in params and isinstance(state.get("random_state"), int):
            # Legacy scikit-learn states store the seed under "random_state".
            params["rng"] = state["random_state"]
        model = cls(
            n_components=int(state.get("n_components", means.shape[0])), **params
        )
        model._weights = np.asarray(state["weights_"], dtype=np.float64)
        model._means = means
        model._covariances = np.asarray(state["covariances_"], dtype=np.float64)
        return model

    @classmethod
    def from_sklearn(cls: type[_GaussianMixtureT], model: Any) -> _GaussianMixtureT:
        """
        Creates a model from a :class:`sklearn.mixture.GaussianMixture`.

        The scikit-learn constructor arguments are translated to their
        equivalents here (``random_state`` becomes ``rng``; ``tol``,
        ``reg_covar``, ``max_iter`` and ``n_init`` map directly).
        ``warm_start``, ``verbose`` and ``verbose_interval`` have no
        counterpart and are ignored. Only ``init_params`` values of
        ``"kmeans"`` and ``"k-means++"`` are accepted — both map to this
        class' greedy k-means++ seeding, the only initialization
        supported.

        If the estimator is fitted, its weights, means and covariances are
        adopted, so the returned model predicts (near-)identically to the
        estimator.

        :param model: A :class:`sklearn.mixture.GaussianMixture` instance,
            fitted or not.
        :return: A model configured (and, if ``model`` is fitted, fitted)
            from the given estimator.
        :raises TypeError: If ``model`` is not a
            :class:`sklearn.mixture.GaussianMixture` instance, or its
            ``random_state`` cannot be translated.
        :raises ValueError: If its covariance type is not diagonal, its
            ``init_params`` is unsupported, or it carries custom initial
            parameters (``weights_init``, ``means_init``,
            ``precisions_init``).
        """
        # Imported lazily: this module is otherwise scikit-learn-free, and the
        # conversion path is the only one that needs the estimator class.
        from sklearn.mixture import GaussianMixture as _SklearnGaussianMixture

        if not isinstance(model, _SklearnGaussianMixture):
            raise TypeError(
                f"{cls.__name__}.from_sklearn expects a "
                "sklearn.mixture.GaussianMixture instance, not "
                f"{type(model).__name__}."
            )
        if model.covariance_type != "diag":
            raise ValueError(
                f"{cls.__name__} only supports covariance_type='diag', "
                f"got {model.covariance_type!r}."
            )
        if model.init_params not in ("kmeans", "k-means++"):
            raise ValueError(
                f"Unsupported scikit-learn init_params {model.init_params!r}; "
                "k-means++ seeding is the only initialization supported."
            )
        for name in ("weights_init", "means_init", "precisions_init"):
            if getattr(model, name) is not None:
                raise ValueError(
                    f"Custom initial parameters ({name}) are not supported."
                )
        params = {
            name: convert(getattr(model, sklearn_name))
            for sklearn_name, (name, convert) in _SKLEARN_PARAM_MAP.items()
        }
        instance = cls(params.pop("n_components"), **params)
        if getattr(model, "means_", None) is not None:
            instance._weights = np.asarray(model.weights_, dtype=np.float64)
            instance._means = np.asarray(model.means_, dtype=np.float64)
            instance._covariances = np.asarray(model.covariances_, dtype=np.float64)
        return instance

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self._n_components}, "
            f"n_init={self._n_init}, fitted={self.is_fitted})"
        )
