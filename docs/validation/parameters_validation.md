# Parameter validation

`pyvisim.utils.validation` checks the arguments of a function before its body
runs. You declare the expected type and value range of each parameter in the
`@validate_params` decorator, and a call that breaks one of them raises
`InvalidParameterError` with a message naming the parameter. Use it to replace 
repetivive guard clases that raise `ValueError` or `TypeError`.

The examples below assume these imports:

```python
import math

import numpy as np

from pyvisim.utils.validation import Param, validate_params
```

> [!TIP]
> You can also call `Param(...).check(name, value)`, which runs the same
check as the decorator without a function around it. See
[Checking a value without a decorator](#checking-a-value-without-a-decorator).

## Quick start

```python
@validate_params(
    n_clusters=Param(int, ge=1),
    thresh=Param(float, ge=0),
    normalize=bool,
)
def fit(n_clusters: int = 8, thresh: float = 1e-5, normalize: bool = True) -> int:
    return n_clusters


fit(16)              # 16
fit(0)               # InvalidParameterError: 'n_clusters' must be >= 1, got 0.
fit(2.5)             # InvalidParameterError: 'n_clusters' must be of type int, got float.
fit(8, thresh=-1.0)  # InvalidParameterError: 'thresh' must be >= 0, got -1.0.
fit(8, normalize=1)  # InvalidParameterError: 'normalize' must be of type bool, got int.
```

A parameter without a keyword in the decorator is not checked.

## Writing a spec

### A bare type


When only the type matters, pass the type itself (which is equal to `Param(type)`):
    
```python
@validate_params(name=str)
def greet(name: str) -> str: ...


greet(3)  # InvalidParameterError: 'name' must be of type str, got int.
```

### Bounds

`Param` takes one keyword per comparison, and the argument has to satisfy every
bound you set:

| Keyword   | The argument must be               | Example spec                           | Message for a bad argument                         |
|-----------|------------------------------------|----------------------------------------|----------------------------------------------------|
| `gt`      | greater than the bound             | `Param(int, gt=0)`                     | `'x' must be > 0, got 0.`                          |
| `ge`      | greater than or equal to the bound | `Param(int, ge=0)`                     | `'x' must be >= 0, got -1.`                        |
| `lt`      | less than the bound                | `Param(int, lt=10)`                    | `'x' must be < 10, got 10.`                        |
| `le`      | less than or equal to the bound    | `Param(int, le=10)`                    | `'x' must be <= 10, got 11.`                       |
| `eq`      | equal to the bound                 | `Param(int, eq=4)`                     | `'x' must be == 4, got 3.`                         |
| `ne`      | different from the bound           | `Param(int, ne=0)`                     | `'x' must be != 0, got 0.`                         |
| `choices` | one of the listed values           | `Param(str, choices=("auto", "full"))` | `'x' must be one of ['auto', 'full'], got 'fast'.` |

A bound left at `None` is not checked. For that reason `None` cannot be a bound
itself. To allow or forbid `None`, use the type, as shown in
[Optional parameters](#optional-parameters).

### Ranges

Combine a lower and an upper bound for a range. `gt` and `lt` leave an end open,
`ge` and `le` close it:

```python
@validate_params(
    margin=Param(float, gt=0, le=2),        # (0, 2]
    lambda_value=Param(float, ge=0, le=1),  # [0, 1]
)
def loss(margin: float = 1.0, lambda_value: float = 0.3) -> None: ...


loss(margin=2)                  # fine
loss(lambda_value=0)            # fine
loss(margin=0)                  # InvalidParameterError: 'margin' must be > 0, got 0.
loss(margin=2.5)                # InvalidParameterError: 'margin' must be <= 2, got 2.5.
```

### Allowed values

`choices` takes a tuple, which limits the argument to one of its values.
This is the enforced version of `Literal["opt1", "opt2"]`:

```python
_SUPPORTED_SOLVERS = ("auto", "full", "covariance_eigh", "arpack")


@validate_params(svd_solver=Param(str, choices=_SUPPORTED_SOLVERS))
def pca(svd_solver: str = "auto") -> None: ...


pca("randomized")
# InvalidParameterError: 'svd_solver' must be one of
# ['auto', 'full', 'covariance_eigh', 'arpack'], got 'randomized'.
```

The options can be of any type that supports `==`:

```python
Param(int, choices=(1, 2, 4, 8)).check("n_bits", 3)
# InvalidParameterError: 'n_bits' must be one of [1, 2, 4, 8], got 3.
```

> [!TIP]
> Pass a tuple even for a single option. A string in its place would turn the
membership test into a substring test.

### A special value next to a range

Some parameters accept one value outside their normal range, such as
`batch_size=-1` for "everything in one batch". `ne` cuts the gap out of a wider
range:

```python
@validate_params(batch_size=Param(int, ge=-1, ne=0))  # -1 or a positive integer
def batches(batch_size: int = 16) -> int: ...


batches(-1)  # fine
batches(0)   # InvalidParameterError: 'batch_size' must be != 0, got 0.
batches(-2)  # InvalidParameterError: 'batch_size' must be >= -1, got -2.
```

The messages don't say what `-1` means, so explain it in the docstring of the
parameter. The `batch_size` of the similarity metrics keeps its own check for
this reason.

### Finite numbers

`float` accepts `inf` and `nan`. An upper bound of `math.inf` rules out
infinity. `NaN` compares false against everything, so it fails every `gt`, `ge`,
`lt` and `le` bound:

```python
@validate_params(alpha=Param(float, ge=0, lt=math.inf))
def expand(alpha: float = 3.0) -> None: ...


expand(math.inf)  # InvalidParameterError: 'alpha' must be < inf, got inf.
expand(math.nan)  # InvalidParameterError: 'alpha' must be >= 0, got nan.
```

A spec with no ordering bound lets `NaN` through, and so does a spec whose only
bound is `ne`.

### Other comparable types

The bounds use Python's comparison operators, so they also work for strings,
dates and any other type that defines them:

```python
import datetime

Param(str, ne="").check("prefix", "")
# InvalidParameterError: 'prefix' must be != '', got ''.

Param(datetime.date, ge=datetime.date(2020, 1, 1)).check(
    "since", datetime.date(2019, 5, 1)
)
# InvalidParameterError: 'since' must be >= datetime.date(2020, 1, 1),
# got datetime.date(2019, 5, 1).
```

## Types

### Numbers

`int` and `float` accept NumPy scalars as well as Python numbers. Python treats
`bool` as a subclass of `int`, but both specs reject it:

| Spec    | Accepts                                              | Rejects                   |
|---------|------------------------------------------------------|---------------------------|
| `int`   | `3`, `np.int64(3)` and other NumPy integers          | `True`, `2.0`, `"3"`      |
| `float` | `0.5`, `1`, `np.float32(0.5)` and other NumPy floats | `False`, `"0.5"`          |
| `bool`  | `True`, `False`                                      | `1`, `np.bool_(True)`     |

```python
Param(int).check("n", np.int64(3))  # fine
Param(int).check("n", True)         # InvalidParameterError: 'n' must be of type int, got bool.
Param(int).check("n", 2.0)          # InvalidParameterError: 'n' must be of type int, got float.
Param(float).check("x", 1)          # fine, an int is a valid float
Param(float).check("x", False)      # InvalidParameterError: 'x' must be of type float, got bool.
```

To accept a `bool` where an `int` is expected, list both types:
`Param(int | bool)`. A NumPy boolean is not a Python `bool`, so convert it with
`bool(...)` before passing it to a `bool` parameter.

The function receives the argument as the caller passed it, so an `int`
parameter can receive an `np.int64`. Convert it in the function body when the
code needs a Python `int`, as the search indexes do with `k = int(k)`.

### Classes

Using classes for type checks also works:

```python
from pyvisim.retrieval.image_store import InMemoryImageEmbeddingStore


@validate_params(store=InMemoryImageEmbeddingStore)
def rerank(store: InMemoryImageEmbeddingStore) -> None: ...


rerank(object())
# InvalidParameterError: 'store' must be of type InMemoryImageEmbeddingStore, got object.
```

Note that the class has to exist when the decorator runs, which is when the
module is imported. For a class the module imports lazily, such as
`torch.nn.Module` in modules that defer the `torch` import, keep an inline
`isinstance` check.

### Several types

Both using a union or a tuple of classes give the same result:

```python
Param(int | str).check("key", 1.5)
# InvalidParameterError: 'key' must be of type int or str, got float.

Param((int, str)).check("key", 1.5)
# InvalidParameterError: 'key' must be of type int or str, got float.
```

### Optional parameters

Add `None` to the type to allow it. A `None` argument skips the bounds and
choices, so `ge=1` below only applies to integers:

```python
@validate_params(num_workers=Param(int | None, ge=1))
def run(num_workers: int | None = None) -> None: ...


run(None)  # fine
run(0)     # InvalidParameterError: 'num_workers' must be >= 1, got 0.
```

Without `None` in the type, `None` fails the type check:

```python
Param(int, ge=1).check("n", None)
# InvalidParameterError: 'n' must be of type int, got NoneType.
```

## Use cases of `validate_params`

### Functions

An argument is checked however the caller passes it, by position or by keyword,
and positional-only and keyword-only parameters work too:

```python
@validate_params(a=Param(int, ge=0), b=Param(int, ge=0), c=Param(int, ge=0))
def kinds(a: int, /, b: int, *, c: int) -> None: ...


kinds(-1, 0, c=0)    # InvalidParameterError: 'a' must be >= 0, got -1.
kinds(0, -1, c=0)    # InvalidParameterError: 'b' must be >= 0, got -1.
kinds(0, b=-1, c=0)  # InvalidParameterError: 'b' must be >= 0, got -1.
kinds(0, 0, c=-1)    # InvalidParameterError: 'c' must be >= 0, got -1.
```

### Methods and constructors

```python
class KMeans:
    @validate_params(n_clusters=Param(int, ge=1), n_init=Param(int, ge=1))
    def __init__(self, n_clusters: int = 256, *, n_init: int = 1) -> None:
        self._n_clusters = n_clusters
        self._n_init = n_init
```

### Property setters

Put `validate_params` below `@<name>.setter`, keyed by the name of the setter's
parameter. A rejected assignment keeps the old value because the setter body
never runs:

```python
class Model:
    def __init__(self) -> None:
        self._n = 1

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    @validate_params(n=Param(int, ge=1))
    def n(self, n: int) -> None:
        self._n = n


model = Model()
model.n = 0  # InvalidParameterError: 'n' must be >= 1, got 0.
model.n      # still 1
```

### Static and class methods

Put `validate_params` below `@staticmethod` or `@classmethod`:

```python
class Factory:
    @staticmethod
    @validate_params(size=Param(int, gt=0))
    def make(size: int) -> int: ...

    @classmethod
    @validate_params(size=Param(int, gt=0))
    def build(cls, size: int) -> "Factory": ...


Factory.make(0)   # InvalidParameterError: 'size' must be > 0, got 0.
Factory.build(0)  # InvalidParameterError: 'size' must be > 0, got 0.
```

### Generators

The arguments of a generator function are checked when it is called, before the
first item is requested:

```python
from collections.abc import Iterator


@validate_params(batch_size=Param(int, ge=1))
def chunks(items: list[int], batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


chunks([1, 2, 3], 0)  # raises here, without iterating
```

### Together with other decorators

Stacked decorators run from the outside in. In
`InMemoryImageEmbeddingStore.retrieve_top_k_similar`, the store first reports a
gallery that was never built and only then looks at the query expansion
settings:

```python
@_requires_built_store
@validate_params(
    expansion_alpha=Param(float, ge=0, lt=math.inf),
    expansion_neighbors=Param(int, ge=1),
)
def retrieve_top_k_similar(
    self,
    query_images: ImageInput,
    k: int = 5,
    *,
    query_expansion: bool = False,
    expansion_alpha: float = 3.0,
    expansion_neighbors: int = 50,
) -> list[list[Candidate]]: ...
```

Placing `validate_params` right above the `def` is the safe default, since it
then reads the signature of the function itself.

## Defaults

The decorator checks every default against its spec once, when the function is
defined, and doesn't check the defaults again on each call. A bad default
therefore fails as soon as the module is imported:

```python
@validate_params(n_clusters=Param(int, ge=1))
def fit(n_clusters: int = 0) -> None: ...
# InvalidParameterError: 'n_clusters' must be >= 1, got 0.
```

A default of `None` needs `None` in the type, as in
[Optional parameters](#optional-parameters).

## Errors raised

The decorator raises on the first problem it finds.

1. Parameters are checked in the order their keywords appear in
   `validate_params`. The order of the arguments in the call does not matter.
2. For each parameter, the type is checked first, then the bounds in the order
   `gt`, `ge`, `lt`, `le`, `eq`, `ne`, and `choices` last.

```python
@validate_params(k1=Param(int, ge=1), k2=Param(int, ge=1))
def rerank(k1: int = 20, k2: int = 6) -> None: ...


rerank(k2=0, k1=0)                              # 'k1' must be >= 1, got 0.
Param(int, ge=1).check("n", -2.5)               # 'n' must be of type int, got float.
Param(int, ge=1, choices=(2, 4)).check("n", 0)  # 'n' must be >= 1, got 0.
```

## Errors

### InvalidParameterError

Every failed check raises `InvalidParameterError`, which inherits
from both `ValueError` and `TypeError`:

```python
from pyvisim._errors import InvalidParameterError

try:
    fit(0)
except ValueError as error:  # TypeError and InvalidParameterError work too
    print(error)             # 'n_clusters' must be >= 1, got 0.
```

In tests, match on the message:

```python
with pytest.raises(ValueError, match="'n_clusters' must be >= 1"):
    fit(0)
```

The messages follow three formats. Several accepted types are joined with
`or`, as in `int or None`.

| Failed check | Message                                                     |
|--------------|-------------------------------------------------------------|
| type         | `'<name>' must be of type <types>, got <type of argument>.` |
| bound        | `'<name>' must be <operator> <bound>, got <argument>.`      |
| choices      | `'<name>' must be one of [<choices>], got <argument>.`      |

In the docstring of a decorated function, list these errors under
`:raises ValueError:`, or under `:raises TypeError:` when the spec is a class
check.

### Mistakes in the decorator

A keyword that names no parameter of the function raises `TypeError` when the
function is defined. So does a keyword for `*args` or `**kwargs`, which collect
several values and cannot be checked as one:

```python
@validate_params(size=int)
def fit(n_clusters: int = 8) -> None: ...
# TypeError: fit() has no named parameter 'size' to validate.


@validate_params(options=dict)
def run(*images, **options) -> None: ...
# TypeError: run() has no named parameter 'options' to validate.
```

A default that breaks its spec raises `InvalidParameterError` at the same
moment. See [Defaults](#defaults).

## Checking a value without a decorator

`Param.check(name, value)` runs the same checks on a single value. Use it for
values that don't arrive as function arguments, such as the entries of a config
dict or of a loaded state. The name only appears in the message:

```python
config = {"n_clusters": "8"}
Param(int, ge=1).check("n_clusters", config["n_clusters"])
# InvalidParameterError: 'n_clusters' must be of type int, got str.
```

## Sharing specs

`Param` is frozen after instantiation, so one instance can be reused anywhere. Keep a
repeated spec in a constant, and unpack a dict of specs when several functions
share a group of parameters:

```python
POSITIVE_INT = Param(int, ge=1)

WINDOW_PARAMS = {
    "sigma": Param(float, gt=0),
    "k1": Param(float, gt=0),
    "k2": Param(float, gt=0),
}


@validate_params(window_size=POSITIVE_INT, **WINDOW_PARAMS)
def ssim(
    window_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03
) -> None: ...


ssim(sigma=0)  # InvalidParameterError: 'sigma' must be > 0, got 0.
```

Every key of the dict has to be a parameter of each function it decorates.

## Signatures, docs and type checkers

The decorated function keeps its name, docstring and signature, so `help()`,
`inspect.signature()` and Sphinx autodoc show the original function. mypy also
sees the original signature and keeps checking calls against it.

```python
import inspect

inspect.signature(fit)  # (n_clusters: int = 8, thresh: float = 1e-05, normalize: bool = True) -> int
fit.__wrapped__         # the undecorated function
```

## Cost

Each call binds the arguments to the signature before checking them. For the
`fit` example in [Quick start](#quick-start), that adds about 2 µs per call on a
laptop. Hence, **leave the decorator off methods that run in a tight inner loop**,
such as per-embedding or per-pixel code.

