.PHONY: test-types test-unit fmt docs

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
