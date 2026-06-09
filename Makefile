.PHONY: test-types

# Strict mypy type-checking
test-types:
	uv run --group types mypy pyvisim/

# Formatting with ruff
fmt:
	uv run --group fmt ruff check --fix .
	uv run --group fmt ruff format .
