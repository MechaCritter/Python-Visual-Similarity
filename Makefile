.PHONY: test-types test-unit test-slow test-notebooks build-ext fmt strip-notebooks check-notebooks docs release-note release-notes

# Regenerate the checked-in Cython C sources and rebuild the editable install.
# --inexact keeps ad-hoc packages in the venv from being pruned.
build-ext:
	uv run --group build cythonize -3 pyvisim/dense/structural/_kernel/_ssim_kernels.pyx
	uv run --group build cythonize -3 pyvisim/features/_vendored/sift/_sift.pyx
	uv run --group build cythonize -3 pyvisim/pixelwise/_kernel/_ssd_kernel.pyx
	uv sync --inexact --reinstall-package pyvisim

# Strict mypy type-checking
test-types:
	uv run --group lint --extra nn mypy pyvisim/

# Unit tests with a terminal coverage report (skips slow, weight-downloading tests)
test-unit:
	uv run --group test --extra nn pytest -m "not slow"

# Test slow tests
test-slow:
	uv run --group test --extra nn pytest -m slow

# Execute every tutorial notebook from scratch and in parallel, like the scheduled
# Tutorials workflow. The executed notebooks are written back in place, so that
# 'make docs' renders their outputs. Strip them again before committing.
# NOTEBOOK_WORKERS limits how many notebooks run at once.
NOTEBOOK_WORKERS = auto
test-notebooks:
	uv run --group tutorials --extra nn pytest -o addopts="" --nbmake --nbmake-timeout=-1 --overwrite -n $(NOTEBOOK_WORKERS) docs/tutorials/notebooks

# Formatting with ruff
fmt:
	uv run --group lint ruff check --fix .
	uv run --group lint ruff format .

# Notebooks are committed without outputs and execution metadata. Their kernel
# metadata goes as well, since the documentation build hashes it to decide
# whether a notebook has to run again.
NOTEBOOKS = $(shell git ls-files --cached --others --exclude-standard '*.ipynb')
NBSTRIPOUT_FLAGS = --keep-id --drop-empty-cells --extra-keys "metadata.kernelspec metadata.language_info"

# Strip every notebook in place before committing it
strip-notebooks:
	uv run --group lint nbstripout $(NBSTRIPOUT_FLAGS) $(NOTEBOOKS)

# Fail if a notebook still carries outputs or metadata (the CI check)
check-notebooks:
	uv run --group lint nbstripout --verify $(NBSTRIPOUT_FLAGS) $(NOTEBOOKS)

# Build the Sphinx HTML documentation for local review, then open
# docs/_build/html/index.html. The tutorial notebooks are not executed but
# rendered as they are on disk, so run 'make test-notebooks' first to see their
# outputs. NB_EXECUTION_MODE=cache executes and caches them like the CI does.
NB_EXECUTION_MODE = off
docs:
	uv run --group docs --group tutorials --extra nn sphinx-build -W -D nb_execution_mode=$(NB_EXECUTION_MODE) -b html docs docs/_build/html

# Create a release note under releasenotes/notes/ for the current change.
# Usage: make release-note NAME=my-change
release-note:
	@test -n "$(NAME)" || echo "Usage: make release-note NAME=my-change" >&2
	@test -n "$(NAME)"
	uv run --group docs reno new $(NAME)

# Render the accumulated release notes for local review
release-notes:
	uv run --group docs reno report --no-show-source --ignore-cache
