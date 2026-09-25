"""
Declarative checks for the arguments of a function.
"""

from .decorators import validate_params
from .param import Param

__all__ = ["Param", "validate_params"]
