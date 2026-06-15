"""
Base classes for the scikit-learn-backed models used by the image encoders.
"""

import abc
from typing import Any, ClassVar, TypeVar

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..typing import FloatNumpyArray

_SklearnModelT = TypeVar("_SklearnModelT", bound="_SklearnModelBase")


class _SklearnModelBase(abc.ABC):
    """
    Base class for models backed by a scikit-learn estimator.

    :param model: Underlying scikit-learn estimator instance.
    """

    _sklearn_class: ClassVar[type[Any]]

    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying estimator has been fitted."""
        try:
            check_is_fitted(self._model)
        except NotFittedError:
            return False
        return True

    def _check_is_fitted(self) -> None:
        """
        Ensures the underlying estimator is fitted before accessing
        fitted-only attributes.

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this attribute."
            )

    @property
    def n_features_in(self) -> int:
        """
        Number of features the fitted estimator expects as input.

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return int(self._model.n_features_in_)

    def fit(self, features: FloatNumpyArray) -> None:
        """
        Fits the underlying estimator on the given feature matrix.

        :param features: Feature matrix of shape (n_samples, n_features).
        """
        self._model.fit(features)

    def _validate_sklearn_model(self) -> None:  # noqa: B027
        """
        Hook for subclasses to validate (or coerce) an estimator that was
        passed in directly via :meth:`_from_sklearn`.
        """

    @classmethod
    def _from_sklearn(cls: type[_SklearnModelT], model: Any) -> _SklearnModelT:
        """
        Creates a model from an existing scikit-learn estimator.

        This is an internal constructor used to adopt pretrained estimators
        (loaded from legacy weight files).

        :param model: Estimator instance of the underlying scikit-learn class.
        :return: A model backed by the given estimator.
        :raises TypeError: If ``model`` is not an instance of the underlying class.
        """
        if not isinstance(model, cls._sklearn_class):
            raise TypeError(
                f"{cls.__name__} can only be created from instances of "
                f"{cls._sklearn_class.__name__}, not {type(model).__name__}."
            )
        instance = cls.__new__(cls)
        _SklearnModelBase.__init__(instance, model)
        instance._validate_sklearn_model()
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"


class ClusteringModelBase(_SklearnModelBase):
    """
    Base class for clustering models.
    """

    @property
    @abc.abstractmethod
    def n_clusters(self) -> int:
        """Number of clusters (or mixture components) of the estimator."""
        raise NotImplementedError
