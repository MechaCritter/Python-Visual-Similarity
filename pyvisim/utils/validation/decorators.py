"""
Decorators that check the arguments a function is called with.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar

from .param import Param, TypeSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")


def validate_params(
    **specs: Param | TypeSpec,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Check the arguments of the decorated function before it runs.

    For more information and examples, see the documentation:
    ``https://github.com/MechaCritter/Python-Visual-Similarity/blob/main/docs/validation/parameters_validation.md``.

    Each keyword names a parameter of the decorated function and maps it to a
    :class:`Param`, or to a bare type as a shorthand for ``Param(type)``. The
    arguments a caller passes are checked on every call. The defaults are
    checked once, when the function is decorated.

    .. code:: python

        @validate_params(n_clusters=Param(int, ge=1), normalize=bool)
        def fit(n_clusters: int = 8, normalize: bool = True) -> None: ...

    :param specs: The spec of each checked parameter, keyed by its name.
    :return: A decorator that adds the checks to a function.
    :raises TypeError: When decorating, if a key is not a named parameter of
        the function.
    :raises InvalidParameterError: When decorating, if a default breaks its
        spec. When calling, if an argument breaks its spec.
    """
    params = {name: _as_param(spec) for name, spec in specs.items()}

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(func)
        _check_names(func, signature, params)
        _check_defaults(signature, params)

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            _check_arguments(signature.bind(*args, **kwargs).arguments, params)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _as_param(spec: Param | TypeSpec) -> Param:
    """Wrap a bare type into a :class:`Param` that only checks the type."""
    return spec if isinstance(spec, Param) else Param(spec)


def _check_names(
    func: Callable[..., Any],
    signature: inspect.Signature,
    params: Mapping[str, Param],
) -> None:
    """Raise if a spec names no parameter, or a ``*args``/``**kwargs`` one."""
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    for name in params:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind in variadic:
            raise TypeError(
                f"{func.__qualname__}() has no named parameter {name!r} to validate."
            )


def _check_defaults(signature: inspect.Signature, params: Mapping[str, Param]) -> None:
    """Check the default of every parameter that has one against its spec."""
    for name, param in params.items():
        default = signature.parameters[name].default
        if default is not inspect.Parameter.empty:
            param.check(name, default)


def _check_arguments(arguments: Mapping[str, Any], params: Mapping[str, Param]) -> None:
    """Check every argument the caller passed against its spec."""
    for name, param in params.items():
        if name in arguments:
            param.check(name, arguments[name])
