.PHONY: test-types test-unit fmt

# Strict mypy type-checking (the 'nn' extra installs torch for type-checking)
test-types:
	uv run --group types --extra nn mypy pyvisim/

# Unit tests with a terminal coverage report (skips slow, weight-downloading tests)
test-unit:
	uv run --group test --extra nn pytest -m "not slow"

# Test slow tests
test-slow:
	uv run --group test --extra nn pytest -m slow
# Formatting with ruff
fmt:
	uv run --group fmt ruff check --fix .
	uv run --group fmt ruff format .
