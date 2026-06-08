.PHONY: test-types

# Strict mypy type-checking
test-types:
	uv run --group types mypy pyvisim/
