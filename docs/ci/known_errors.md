# Known type-check issues

## `make test-types` fails on `numpy/__init__.pyi` with NumPy >= 2.3

`[tool.mypy]` in `pyproject.toml` sets `python_version = "3.10"`, the Python
version CI runs on. The type stubs shipped with NumPy 2.3 and newer use the
PEP 695 `type` statement, which mypy rejects for a 3.10 target:

```
numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater  [syntax]
Found 1 error in 1 file (errors prevented further checking)
```

Since the error is raised while the stub is parsed, nothing under `pyvisim/`
gets checked in that run. The failure comes from the NumPy version installed in your virtual environment, so your branch should be fine. CI does not hit it
because NumPy 2.3 and newer no longer support Python 3.10, so CI resolves NumPy 2.2.x, whose stubs contain no `type` statements.

To type-check locally in such an environment, point mypy at a newer target:

```bash
uv run --group lint --extra nn mypy --python-version 3.14 pyvisim/
```

Keep in mind that this only approximates CI. The NumPy 2.2.x stubs used in CI
type the result of many array operations as `Any`, which `strict` reports as
`no-any-return`. The newer stubs infer a concrete array type instead and
report the `cast(...)` that silences CI as `redundant-cast`. So a green run
with the workaround can still be followed by a `no-any-return` error in CI,
and vice versa. For a faithful replica, create a throwaway environment that
matches CI and run it from the repository root so that `pyproject.toml` is
picked up:

```bash
uv venv --python 3.10 /tmp/py310
VIRTUAL_ENV=/tmp/py310 uv pip install "numpy<2.3" mypy
/tmp/py310/bin/mypy --strict --warn-return-any --no-pretty path/to/changed_file.py
```

Running it on the whole package also needs the `nn` extra in that
environment, which pulls a large `torch` wheel for CPython 3.10, so it is
usually run on the changed files only.

## `pyvisim/classic/fisher_vector.py`: redundant cast to `Float64NumpyArray`

With the workaround above, one pre-existing error is reported:

```
pyvisim/classic/fisher_vector.py: error: Redundant cast to "ndarray[tuple[Any, ...], dtype[float64]]"  [redundant-cast]
```

The NumPy 2.2.x stubs used in CI type the result of the division as `Any`, so
the cast is needed there to avoid `no-any-return`. Only the newer stubs call
it redundant. The error is expected with the workaround and is not caused by
your change. Removing the cast would currently break CI!
