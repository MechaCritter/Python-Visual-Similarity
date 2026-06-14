.PHONY: test-types test-unit fmt

# Strict mypy type-checking
test-types:
	uv run --group types mypy pyvisim/

# Unit tests with a terminal coverage report (skips slow, weight-downloading tests)
test-unit:
	uv run --group test pytest -m "not slow"

# Test slow tests
test-slow:
	uv run --group test pytest -m slow
# Formatting with ruff
fmt:
	uv run --group fmt ruff check --fix .
	uv run --group fmt ruff format .
