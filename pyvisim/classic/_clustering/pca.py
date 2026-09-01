"""Principal Component Analysis model used by the image embedders."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from scipy.linalg import eigh, svd
from scipy.sparse.linalg import svds

from ..._errors import NotFittedError
from ...typing import Float64NumpyArray, FloatNumpyArray
from ._base_clustering import _decode, _embed
from .kmeans import _rng_from_sklearn

_PCAT = TypeVar("_PCAT", bound="PCA")

_SUPPORTED_SOLVERS = ("auto", "full", "covariance_eigh", "arpack")
_INIT_PARAM_NAMES = ("whiten", "svd_solver", "tol", "rng")


def _n_components_from_sklearn(n_components: Any) -> int:
    """
    Translates a scikit-learn ``n_components`` value into an integer count.

    :param n_components: The scikit-learn ``n_components`` constructor argument.
    :return: The number of components to keep.
    :raises TypeError: If ``n_components`` is not an integer (scikit-learn's
        ``None``, float and ``"mle"`` variants resolve the count from the
        training data and cannot be mapped up front).
    """
    if isinstance(n_components, (int, np.integer)) and not isinstance(
        n_components, bool
    ):
        return int(n_components)
    raise TypeError(
        f"Unsupported scikit-learn n_components {n_components!r}; only an "
        "integer component count can be mapped."
    )


def _solver_from_sklearn(svd_solver: Any) -> str:
    """
    Translates a scikit-learn ``svd_solver`` into a supported solver name.

    :param svd_solver: The scikit-learn ``svd_solver`` constructor argument.
    :return: One of the supported solver names. ``"randomized"`` (which has
        no counterpart here) falls back to ``"auto"``.
    """
    if svd_solver in _SUPPORTED_SOLVERS:
        return str(svd_solver)
    return "auto"


_SKLEARN_PARAM_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "n_components": ("n_components", _n_components_from_sklearn),
    "whiten": ("whiten", bool),
    "svd_solver": ("svd_solver", _solver_from_sklearn),
    "tol": ("tol", float),
    "random_state": ("rng", _rng_from_sklearn),
}


def _flip_signs(components: Float64NumpyArray) -> Float64NumpyArray:
    """
    Makes the component signs deterministic.

    The sign of an SVD/eigendecomposition factor pair is arbitrary, so the
    same data can yield components flipped between runs, solvers or BLAS
    builds. Each component is flipped so that its entry with the largest
    absolute value is positive.

    :param components: Component matrix of shape (n_components, n_features).
    :return: The sign-adjusted components (modified in place and returned).
    """
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), largest])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    return components


class PCA:
    """
    Principal Component Analysis model, used by the image embedders to
    reduce the dimensionality of local features.

    Supported SVD solvers are:

    - ``"full"``: economy SVD of the centered features.
    - ``"covariance_eigh"``: eigendecomposition of the (n_features,
      n_features) covariance matrix. The covariance is centered *after* the
      Gram product ``X.T @ X``, so no centered copy of the data is ever
      materialised — the method of choice for tall-and-skinny data such as
      millions of local SIFT descriptors.
    - ``"arpack"``: truncated sparse SVD computing only the requested
      ``n_components`` singular triplets; worthwhile when ``n_components``
      is much smaller than both matrix dimensions. Requires
      ``n_components < min(n_samples, n_features)``.
    - ``"auto"``: picks one of the above at fit time:
      ``"covariance_eigh"`` when ``n_features <= 1000`` and
      ``n_samples >= 10 * n_features``; else ``"full"`` when
      ``max(n_samples, n_features) <= 500``; else ``"arpack"`` when
      ``n_components < 0.8 * min(n_samples, n_features)``; ``"full"``
      otherwise.

    Component signs are made deterministic by flipping each component so
    that its largest-magnitude entry is positive, so all solvers produce
    comparable output.

    :meth:`transform` always centers with the **training** mean stored at
    fit time — never the mean of the batch being transformed — so a single
    query image is projected exactly like the training corpus was. With
    ``whiten=True``, components whose standard deviation falls below
    machine epsilon (rank-deficient training data) are divided by epsilon
    instead, keeping the output finite.

    :param n_components: Number of components to keep. Must be at most
        ``min(n_samples, n_features)`` of the training data.
    :param whiten: Whether to scale the projected features to unit
        component-wise variance.
    :param svd_solver: One of ``"auto"``, ``"full"``, ``"covariance_eigh"``
        or ``"arpack"`` (see above).
    :param tol: Convergence tolerance of the ``"arpack"`` solver (0 means
        machine precision). Ignored by the other solvers.
    :param rng: Seed (int) or :class:`numpy.random.Generator` for the
        ``"arpack"`` solver's starting vector, making its iteration
        deterministic. Ignored by the other solvers.
    :raises ValueError: If ``n_components`` is not positive, ``tol`` is
        negative, or ``svd_solver`` is not one of the supported names.
    """

    def __init__(
        self,
        n_components: int,
        *,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be positive, got {n_components}.")
        if svd_solver not in _SUPPORTED_SOLVERS:
            raise ValueError(
                f"svd_solver must be one of {_SUPPORTED_SOLVERS}, got {svd_solver!r}."
            )
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        self._n_components = int(n_components)
        self._whiten = bool(whiten)
        self._svd_solver = str(svd_solver)
        self._tol = float(tol)
        self._rng = rng
        self._components: Float64NumpyArray | None = None
        self._mean: Float64NumpyArray | None = None
        self._explained_variance: Float64NumpyArray | None = None
        self._explained_variance_ratio: Float64NumpyArray | None = None
        self._singular_values: Float64NumpyArray | None = None
        self._noise_variance = 0.0
        self._n_samples = 0
        self._fit_svd_solver: str | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted (components are stored)."""
        return self._components is not None

    def _check_is_fitted(self) -> None:
        """
        Ensures the model is fitted before accessing fitted-only attributes.

        :raises NotFittedError: If the model is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this attribute."
            )

    @property
    def n_components(self) -> int:
        """Number of components of the fitted PCA."""
        self._check_is_fitted()
        return self._n_components

    @property
    def n_features_in(self) -> int:
        """Number of features the fitted model expects as input."""
        return int(self.components.shape[1])

    @property
    def whiten(self) -> bool:
        """Whether the projected features are scaled to unit variance."""
        return self._whiten

    @property
    def components(self) -> Float64NumpyArray:
        """
        Principal components, shape (n_components, n_features), sorted by
        descending explained variance.
        """
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._components)

    @property
    def mean(self) -> Float64NumpyArray:
        """Per-feature mean of the training data, shape (n_features,)."""
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._mean)

    @property
    def explained_variance(self) -> Float64NumpyArray:
        """Variance explained by each component, shape (n_components,)."""
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._explained_variance)

    @property
    def explained_variance_ratio(self) -> Float64NumpyArray:
        """
        Fraction of the total variance explained by each component,
        shape (n_components,).
        """
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._explained_variance_ratio)

    @property
    def singular_values(self) -> Float64NumpyArray:
        """
        Singular values of the centered training data corresponding to each
        component, shape (n_components,).
        """
        self._check_is_fitted()
        return cast(Float64NumpyArray, self._singular_values)

    @property
    def noise_variance(self) -> float:
        """
        Mean variance of the discarded components (the probabilistic-PCA
        noise estimate); 0 when nothing is discarded.
        """
        self._check_is_fitted()
        return self._noise_variance

    @property
    def fitted_svd_solver(self) -> str:
        """Solver that actually ran at fit time (``"auto"`` resolved)."""
        self._check_is_fitted()
        return cast(str, self._fit_svd_solver)

    @staticmethod
    def _validate_features(
        features: FloatNumpyArray, *, n_features: int | None = None
    ) -> Float64NumpyArray:
        """
        Coerces a feature matrix to ``float64`` and validates its shape.
        """
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

    def _resolve_solver(self, n_samples: int, n_features: int) -> str:
        """
        Resolves ``svd_solver="auto"`` for the given training shape.

        :param n_samples: Number of training samples.
        :param n_features: Number of training features.
        :return: The solver to run.
        """
        if self._svd_solver != "auto":
            return self._svd_solver
        if n_features <= 1_000 and n_samples >= 10 * n_features:
            return "covariance_eigh"
        if max(n_samples, n_features) <= 500:
            return "full"
        if self._n_components < 0.8 * min(n_samples, n_features):
            return "arpack"
        return "full"

    def _decompose_full(
        self, data: Float64NumpyArray, mean: Float64NumpyArray
    ) -> tuple[Float64NumpyArray, Float64NumpyArray, Float64NumpyArray, float]:
        """
        Decomposes the centered data with an economy SVD.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param mean: Per-feature mean of ``data``, shape (n_features,).
        :return: ``(spectrum, singular_values, components, total_variance)``
            where ``spectrum`` holds all ``min(n_samples, n_features)``
            variances in descending order.
        """
        _, singular_values, components = svd(data - mean, full_matrices=False)
        spectrum = singular_values**2 / (data.shape[0] - 1)
        return spectrum, singular_values, components, float(spectrum.sum())

    def _decompose_covariance(
        self, data: Float64NumpyArray, mean: Float64NumpyArray
    ) -> tuple[Float64NumpyArray, Float64NumpyArray, Float64NumpyArray, float]:
        """
        Decomposes the covariance matrix of the data.

        The covariance is centered after the Gram product
        (``X.T @ X - n * mean mean^T``), so no centered (n_samples,
        n_features) copy of the data is allocated. The eigenvalues of the
        nearly-PSD covariance can come out slightly negative through
        floating-point cancellation; they are clipped at zero before the
        square root that recovers the singular values.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param mean: Per-feature mean of ``data``, shape (n_features,).
        :return: ``(spectrum, singular_values, components, total_variance)``
            where ``spectrum`` holds all ``n_features`` variances in
            descending order.
        """
        n_samples = data.shape[0]
        covariance = data.T @ data
        covariance -= n_samples * np.outer(mean, mean)
        covariance /= n_samples - 1
        eigenvalues, eigenvectors = eigh(covariance)
        spectrum = np.maximum(eigenvalues[::-1], 0.0)
        components = np.asarray(eigenvectors[:, ::-1].T)
        singular_values = np.sqrt(spectrum * (n_samples - 1))
        return spectrum, singular_values, components, float(spectrum.sum())

    def _decompose_truncated(
        self, data: Float64NumpyArray, mean: Float64NumpyArray
    ) -> tuple[Float64NumpyArray, Float64NumpyArray, Float64NumpyArray, float]:
        """
        Computes only the leading ``n_components`` singular triplets.

        The starting vector is drawn from ``rng``, making the otherwise
        randomly-seeded Lanczos iteration reproducible. The singular values
        come out in ascending order, so the outputs are reversed.
        The total variance (which the truncated spectrum cannot provide)
        is computed directly from the centered data.

        :param data: Feature matrix of shape (n_samples, n_features).
        :param mean: Per-feature mean of ``data``, shape (n_features,).
        :return: ``(spectrum, singular_values, components, total_variance)``
            where ``spectrum`` holds the ``n_components`` leading variances
            in descending order.
        """
        n_samples = data.shape[0]
        centered = data - mean
        generator = np.random.default_rng(self._rng)
        v0 = generator.uniform(-1.0, 1.0, min(centered.shape))
        _, singular_values, components = svds(
            centered, k=self._n_components, tol=self._tol, v0=v0
        )
        singular_values = singular_values[::-1]
        components = np.asarray(components[::-1])
        spectrum = singular_values**2 / (n_samples - 1)
        total_variance = float(
            np.einsum("ij,ij->", centered, centered) / (n_samples - 1)
        )
        return spectrum, singular_values, components, total_variance

    def fit(self, features: FloatNumpyArray) -> None:
        """
        Learns the principal components from the given feature matrix.

        The solver resolved from ``svd_solver`` decomposes the centered
        features (see the class docstring); the components, their signs
        made deterministic, are stored on the model together with the
        training mean and the explained-variance spectrum.

        :param features: Feature matrix of shape (n_samples, n_features).
        :raises ValueError: If the matrix is not 2-dimensional, has fewer
            than 2 samples (the sample variance is undefined), or
            ``n_components`` exceeds what its shape and the solver allow.
        """
        data = self._validate_features(features)
        n_samples, n_features = data.shape
        if n_samples < 2:
            raise ValueError(
                f"PCA requires at least 2 samples to estimate variances, "
                f"got {n_samples}."
            )
        max_components = min(n_samples, n_features)
        if self._n_components > max_components:
            raise ValueError(
                f"n_components={self._n_components} must be at most "
                f"min(n_samples, n_features)={max_components}."
            )
        solver = self._resolve_solver(n_samples, n_features)
        if solver == "arpack" and self._n_components == max_components:
            raise ValueError(
                f"n_components={self._n_components} must be strictly less "
                f"than min(n_samples, n_features)={max_components} with "
                "svd_solver='arpack'."
            )

        mean = np.asarray(data.mean(axis=0))
        decompose = {
            "full": self._decompose_full,
            "covariance_eigh": self._decompose_covariance,
            "arpack": self._decompose_truncated,
        }[solver]
        spectrum, singular_values, components, total_variance = decompose(data, mean)
        components = _flip_signs(components)

        keep = self._n_components
        self._components = np.ascontiguousarray(components[:keep])
        self._explained_variance = np.ascontiguousarray(spectrum[:keep])
        self._explained_variance_ratio = np.asarray(
            self._explained_variance / total_variance
        )
        self._singular_values = np.ascontiguousarray(singular_values[:keep])
        if keep < max_components:
            if solver == "arpack":
                # The truncated spectrum has no discarded tail to average;
                # spread the unexplained variance over the discarded rank.
                self._noise_variance = float(
                    (total_variance - self._explained_variance.sum())
                    / (max_components - keep)
                )
            else:
                self._noise_variance = float(np.mean(spectrum[keep:]))
        else:
            self._noise_variance = 0.0
        self._mean = mean
        self._n_samples = n_samples
        self._fit_svd_solver = solver

    def transform(self, features: FloatNumpyArray) -> Float64NumpyArray:
        """
        Projects the given features onto the principal components.

        The features are centered with the **training** mean stored at fit
        time, applied after the projection
        (``X @ V.T - mean @ V.T``) so the input matrix is not copied. With
        ``whiten=True``, each projected component is divided by its
        standard deviation.

        :param features: Feature matrix of shape (n_samples, n_features).
        :return: Reduced features of shape (n_samples, n_components).
        :raises NotFittedError: If the model is not fitted.
        :raises ValueError: If the feature count does not match the
            fitted model.
        """
        components = self.components
        data = self._validate_features(features, n_features=components.shape[1])
        transformed = data @ components.T
        transformed -= self.mean @ components.T
        if self._whiten:
            scale = np.sqrt(self.explained_variance)
            eps = np.finfo(scale.dtype).eps
            scale[scale < eps] = eps
            transformed /= scale
        return np.asarray(transformed)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the fitted model into a JSON-safe dictionary.

        **NOTE**: ``rng`` is stored only when it is an integer seed; a
        :class:`numpy.random.Generator` is stateful and not JSON-safe.

        :return: A dictionary describing the model class and its fitted state.
        :raises NotFittedError: If the model is not fitted.
        """
        state = {
            "components_": self.components,
            "mean_": self.mean,
            "explained_variance_": self.explained_variance,
            "explained_variance_ratio_": self.explained_variance_ratio,
            "singular_values_": self.singular_values,
            "noise_variance_": self._noise_variance,
            "n_samples_": self._n_samples,
            "n_components": self._n_components,
            "n_components_": self._n_components,
            "n_features_in_": self.n_features_in,
            "whiten": self._whiten,
            "svd_solver": self._svd_solver,
            "_fit_svd_solver": self._fit_svd_solver,
            "tol": self._tol,
            "rng": self._rng if isinstance(self._rng, int) else None,
        }
        return {
            "__class__": type(self).__name__,
            "__module__": type(self).__module__,
            "state": _embed(state),
        }

    @classmethod
    def from_dict(cls: type[_PCAT], data: dict[str, Any]) -> _PCAT:
        """
        Rebuilds a model from a dictionary produced by :meth:`to_dict`.

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
        for key in ("components_", "mean_", "explained_variance_"):
            if key not in state:
                raise ValueError(f"Malformed {cls.__name__} state; missing {key!r}.")
        components = np.asarray(state["components_"], dtype=np.float64)
        explained_variance = np.asarray(state["explained_variance_"], dtype=np.float64)
        n_samples = int(state.get("n_samples_", 0))

        params = {name: state[name] for name in _INIT_PARAM_NAMES if name in state}
        if params.get("svd_solver") not in (None, *_SUPPORTED_SOLVERS):
            params["svd_solver"] = "auto"
        if "rng" not in params and isinstance(state.get("random_state"), int):
            # Legacy scikit-learn states store the seed under "random_state".
            params["rng"] = state["random_state"]
        model = cls(
            int(state.get("n_components_", components.shape[0])),
            **params,
        )
        model._components = components
        model._mean = np.asarray(state["mean_"], dtype=np.float64)
        model._explained_variance = explained_variance
        model._explained_variance_ratio = np.asarray(
            state.get(
                "explained_variance_ratio_",
                explained_variance / explained_variance.sum(),
            ),
            dtype=np.float64,
        )
        model._singular_values = np.asarray(
            state.get(
                "singular_values_",
                np.sqrt(explained_variance * max(n_samples - 1, 1)),
            ),
            dtype=np.float64,
        )
        model._noise_variance = float(state.get("noise_variance_", 0.0))
        model._n_samples = n_samples
        fitted_solver = state.get("_fit_svd_solver")
        model._fit_svd_solver = (
            str(fitted_solver) if fitted_solver is not None else model._svd_solver
        )
        return model

    @classmethod
    def from_sklearn(cls: type[_PCAT], model: Any) -> _PCAT:
        """
        Creates a model from a :class:`sklearn.decomposition.PCA` estimator.

        :param model: A :class:`sklearn.decomposition.PCA` instance,
            fitted or not.
        :return: A model configured (and, if ``model`` is fitted, fitted)
            from the given estimator.
        :raises TypeError: If ``model`` is not a
            :class:`sklearn.decomposition.PCA` instance, or its
            ``n_components`` / ``random_state`` cannot be translated.
        """
        # Imported lazily: this module is otherwise scikit-learn-free, and the
        # conversion path is the only one that needs the estimator class.
        from sklearn.decomposition import PCA as _SklearnPCA

        if not isinstance(model, _SklearnPCA):
            raise TypeError(
                f"{cls.__name__}.from_sklearn expects a "
                f"sklearn.decomposition.PCA instance, not {type(model).__name__}."
            )
        params = {
            name: convert(getattr(model, sklearn_name))
            for sklearn_name, (name, convert) in _SKLEARN_PARAM_MAP.items()
        }
        instance = cls(params.pop("n_components"), **params)
        if getattr(model, "components_", None) is not None:
            instance._components = np.asarray(model.components_, dtype=np.float64)
            instance._mean = np.asarray(model.mean_, dtype=np.float64)
            instance._explained_variance = np.asarray(
                model.explained_variance_, dtype=np.float64
            )
            instance._explained_variance_ratio = np.asarray(
                model.explained_variance_ratio_, dtype=np.float64
            )
            instance._singular_values = np.asarray(
                model.singular_values_, dtype=np.float64
            )
            instance._noise_variance = float(model.noise_variance_)
            instance._n_samples = int(model.n_samples_)
            instance._fit_svd_solver = str(
                getattr(model, "_fit_svd_solver", model.svd_solver)
            )
        return instance

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self._n_components}, "
            f"whiten={self._whiten}, svd_solver={self._svd_solver!r}, "
            f"fitted={self.is_fitted})"
        )
