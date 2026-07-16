.PHONY: test-types test-unit fmt docs

# Regenerate the checked-in Cython C sources and rebuild the editable install.
# --inexact keeps ad-hoc packages in the venv from being pruned.
build-ext:
	uv run --group build cythonize -3 pyvisim/structural/_kernel/_ssim_kernels.pyx
	uv sync --inexact --reinstall-package pyvisim

# Strict mypy type-checking ('nn' installs torch; 'search' installs faiss)
test-types:
	uv run --group types --extra nn --extra search mypy pyvisim/

# Unit tests with a terminal coverage report (skips slow, weight-downloading tests)
test-unit:
	uv run --group test --extra nn --extra search pytest -m "not slow"

# Test slow tests
test-slow:
	uv run --group test --extra nn --extra search pytest -m slow
# Formatting with ruff
fmt:
	uv run --group fmt ruff check --fix .
	uv run --group fmt ruff format .

# Build the Sphinx HTML documentation for local review (same flags as CI);
# open docs/sphinx/_build/html/index.html afterwards
docs:
	uv run --group docs --extra nn --extra search sphinx-build -W -b html docs/sphinx docs/sphinx/_build/html
