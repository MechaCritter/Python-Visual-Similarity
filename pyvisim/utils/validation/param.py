"""
Expected type and value range of a single function parameter.
"""

from __future__ import annotations

import dataclasses
import numbers
import operator
from collections.abc import Callable
from types import NoneType, UnionType
from typing import Any, TypeAlias, get_args

from ..._errors import InvalidParameterError

# A class, a tuple of classes or a union such as ``int | None``.
TypeSpec: TypeAlias = type | UnionType | tuple[type, ...]

# Abstract base classes checked in place of the built-in numeric types, so
# NumPy scalars pass as well. ``bool`` is still rejected for both.
_NUMERIC_TOWER: dict[type, type] = {int: numbers.Integral, float: numbers.Real}

# Bound field, the operator shown in error messages and the comparison the
# argument has to satisfy against the bound.
_COMPARISONS: tuple[tuple[str, str, Callable[[Any, Any], Any]], ...] = (
    ("gt", ">", operator.gt),
    ("ge", ">=", operator.ge),
    ("lt", "<", operator.lt),
    ("le", "<=", operator.le),
    ("eq", "==", operator.eq),
    ("ne", "!=", operator.ne),
)


@dataclasses.dataclass(frozen=True)
class Param:
    """
    Expected type and value range of one parameter, checked by
    :func:`~pyvisim.utils.validation.validate_params`.

    ``int`` accepts any integral number and ``float`` any real number, NumPy
    scalars included, but neither accepts a ``bool``. A ``None`` argument
    that the type allows skips the bound and choice checks. A ``NaN``
    argument fails every ordering bound.

    :param expected_type: The accepted class, tuple of classes or union, for
        example ``int | None``.
    :param gt: If not ``None``, the argument must be greater than this.
    :param ge: If not ``None``, the argument must be greater than or equal
        to this.
    :param lt: If not ``None``, the argument must be less than this.
    :param le: If not ``None``, the argument must be less than or equal to
        this.
    :param eq: If not ``None``, the argument must equal this.
    :param ne: If not ``None``, the argument must differ from this.
    :param choices: If not ``None``, the argument must be one of these values.
    """

    expected_type: TypeSpec
    gt: Any = None
    ge: Any = None
    lt: Any = None
    le: Any = None
    eq: Any = None
    ne: Any = None
    choices: tuple[Any, ...] | None = None

    def check(self, name: str, value: Any) -> None:
        """
        Check one argument against this spec.

        :param name: Name of the parameter, used in the error message.
        :param value: The argument passed for the parameter.
        :raises InvalidParameterError: If ``value`` has the wrong type, breaks
            a bound or is not one of the choices.
        """
        self._check_type(name, value)
        if value is None:
            return
        self._check_bounds(name, value)
        self._check_choices(name, value)

    @property
    def _classes(self) -> tuple[type, ...]:
        """The accepted classes, with any union or tuple flattened."""
        if isinstance(self.expected_type, UnionType):
            return get_args(self.expected_type)
        if isinstance(self.expected_type, tuple):
            return self.expected_type
        return (self.expected_type,)

    def _check_type(self, name: str, value: Any) -> None:
        """Raise if ``value`` is not an instance of any accepted class."""
        if any(_is_instance(value, cls) for cls in self._classes):
            return
        expected = " or ".join(_type_name(cls) for cls in self._classes)
        raise InvalidParameterError(
            f"'{name}' must be of type {expected}, got {type(value).__name__}."
        )

    def _check_bounds(self, name: str, value: Any) -> None:
        """Raise if ``value`` breaks any of the set bounds."""
        for field, symbol, holds in _COMPARISONS:
            bound = getattr(self, field)
            if bound is not None and not holds(value, bound):
                raise InvalidParameterError(
                    f"'{name}' must be {symbol} {bound!r}, got {value!r}."
                )

    def _check_choices(self, name: str, value: Any) -> None:
        """Raise if choices are set and ``value`` is not one of them."""
        if self.choices is None or value in self.choices:
            return
        raise InvalidParameterError(
            f"'{name}' must be one of {list(self.choices)!r}, got {value!r}."
        )


def _is_instance(value: Any, cls: type) -> bool:
    """
    Report whether ``value`` is an instance of ``cls``.

    The built-in ``int`` and ``float`` are widened to their abstract numeric
    base classes, which exclude ``bool`` here.

    :param value: The argument to test.
    :param cls: One accepted class.
    :return: ``True`` if ``value`` passes as an instance of ``cls``.
    """
    numeric = _NUMERIC_TOWER.get(cls)
    if numeric is None:
        return isinstance(value, cls)
    return isinstance(value, numeric) and not isinstance(value, bool)


def _type_name(cls: type) -> str:
    """Name of ``cls`` as shown in error messages."""
    return "None" if cls is NoneType else cls.__name__
