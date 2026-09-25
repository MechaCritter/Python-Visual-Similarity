"""Tests for :func:`pyvisim.utils.validation.validate_params` and :class:`Param`."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

from pyvisim._errors import InvalidParameterError
from pyvisim.utils.validation import Param, validate_params


@validate_params(
    count=Param(int, ge=1),
    ratio=Param(float, gt=0, le=1),
    mode=Param(str, choices=("fast", "exact")),
    limit=Param(int | None, lt=10),
    flag=bool,
)
def _configure(
    count: int,
    ratio: float = 0.5,
    mode: str = "fast",
    limit: int | None = None,
    flag: bool = False,
) -> int:
    """Return ``count`` once every argument passed its spec."""
    return count


class _Counter:
    """Holds a count whose setter and method are both validated."""

    def __init__(self) -> None:
        self._count = 1

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    @validate_params(count=Param(int, ge=1))
    def count(self, count: int) -> None:
        self._count = count

    @validate_params(step=Param(int, ne=0))
    def advance(self, step: int) -> int:
        return self._count + step


# Accepted arguments


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 1},
        {"count": np.int64(3), "ratio": np.float32(1.0)},
        {"count": 2, "ratio": 1, "mode": "exact", "limit": 9, "flag": True},
        {"count": 2, "limit": None},
    ],
)
def test_valid_arguments_reach_the_function(kwargs: dict[str, Any]) -> None:
    """Arguments inside their specs are passed through unchanged."""
    assert _configure(**kwargs) == kwargs["count"]


def test_positional_arguments_are_checked() -> None:
    """Arguments bound by position are checked like keyword arguments."""
    with pytest.raises(InvalidParameterError, match="'ratio' must be > 0"):
        _configure(1, 0.0)


# Rejected types


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 2.0}, "'count' must be of type int, got float."),
        ({"count": True}, "'count' must be of type int, got bool."),
        ({"count": 1, "ratio": "0.5"}, "'ratio' must be of type float, got str."),
        ({"count": 1, "ratio": False}, "'ratio' must be of type float, got bool."),
        ({"count": 1, "limit": 1.5}, "'limit' must be of type int or None, got float."),
        ({"count": 1, "flag": 1}, "'flag' must be of type bool, got int."),
    ],
)
def test_wrong_types_raise(kwargs: dict[str, Any], message: str) -> None:
    """``int`` and ``float`` reject ``bool`` and every non-numeric type."""
    with pytest.raises(InvalidParameterError, match=f"^{message}$"):
        _configure(**kwargs)


# Rejected values


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "'count' must be >= 1, got 0."),
        ({"count": 1, "ratio": 0.0}, "'ratio' must be > 0, got 0.0."),
        ({"count": 1, "ratio": 1.5}, "'ratio' must be <= 1, got 1.5."),
        ({"count": 1, "limit": 10}, "'limit' must be < 10, got 10."),
        (
            {"count": 1, "mode": "slow"},
            "'mode' must be one of ['fast', 'exact'], got 'slow'.",
        ),
    ],
)
def test_out_of_range_values_raise(kwargs: dict[str, Any], message: str) -> None:
    """Every bound and the choices produce a message naming the parameter."""
    with pytest.raises(InvalidParameterError) as error:
        _configure(**kwargs)
    assert str(error.value) == message


def test_nan_fails_an_ordering_bound() -> None:
    """``NaN`` compares false against every bound and is therefore rejected."""
    with pytest.raises(InvalidParameterError, match="'ratio' must be > 0"):
        _configure(1, ratio=math.nan)


@pytest.mark.parametrize(("bound", "value"), [("eq", 2), ("ne", 3)])
def test_equality_bounds(bound: str, value: int) -> None:
    """``eq`` and ``ne`` compare the argument with ``==`` and ``!=``."""
    param = Param(int, **{bound: 3})
    with pytest.raises(InvalidParameterError, match="'x' must be"):
        param.check("x", value)


def test_error_is_both_a_value_and_a_type_error() -> None:
    """Existing ``except ValueError`` and ``except TypeError`` clauses keep working."""
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidParameterError, TypeError)


# Methods and properties


def test_property_setter_is_checked_and_keeps_the_old_value() -> None:
    """A rejected assignment leaves the property unchanged."""
    counter = _Counter()
    with pytest.raises(InvalidParameterError, match="'count' must be >= 1"):
        counter.count = 0
    assert counter.count == 1


def test_method_arguments_are_checked() -> None:
    """``self`` is bound but only the named parameter is checked."""
    counter = _Counter()
    assert counter.advance(2) == 3
    with pytest.raises(InvalidParameterError, match="'step' must be != 0"):
        counter.advance(0)


def test_signature_and_docstring_are_preserved() -> None:
    """The wrapper looks like the function it decorates."""
    assert list(inspect.signature(_configure).parameters) == [
        "count",
        "ratio",
        "mode",
        "limit",
        "flag",
    ]
    assert _configure.__doc__ == "Return ``count`` once every argument passed its spec."


# Decoration errors


def test_unknown_parameter_name_raises_when_decorating() -> None:
    """A spec for a parameter the function does not have is a programming error."""
    with pytest.raises(TypeError, match="has no named parameter 'size'"):

        @validate_params(size=int)
        def _function(count: int) -> None: ...


def test_variadic_parameter_raises_when_decorating() -> None:
    """``*args`` and ``**kwargs`` hold several arguments and cannot be checked."""
    with pytest.raises(TypeError, match="has no named parameter 'kwargs'"):

        @validate_params(kwargs=dict)
        def _function(**kwargs: Any) -> None: ...


def test_invalid_default_raises_when_decorating() -> None:
    """A default outside its spec is reported when the function is defined."""
    with pytest.raises(InvalidParameterError, match="'count' must be >= 1, got 0."):

        @validate_params(count=Param(int, ge=1))
        def _function(count: int = 0) -> None: ...
