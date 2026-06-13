# Contributing to pyvisim

Thank you for your interest! Contributions of all kinds are welcome.


## PR TODO list - your first PR

To understand how the library is structured as well the technical details before diving in, you can first read the [developer documentation](docs/overview.md), and/or you can also read the docstrings of the modules and classes that you are working on.

Use this checklist to stay on track for your first code PR:

- **Clone this repository**: see [Set up developer environment](#set-up-developer-environment) section.
- **Check out the coding style**: see [Code style](#code-style) section.
- **Run tests**: run `make test-types` and `make fmt` before you make a PR.
- **Open a PR** on GitHub.

## Using AI to contribute

I know, we all use Claude/Codex/OpenClaw and co. to help us write code faster. I am no exception. Just make sure that you review the generated code carefully before you make your PR.

> [!IMPORTANT]
> It is not difficult to detect an AI-generated PR that was not reviewed at all, and I will have to reject such PRs immediately because it shows you did not take time checking what the AI wrote 🙂.

Please keep pull requests focused - **only one feature or fix per PR**! That would
make review faster.

## Reporting issues

Open an issue on [GitHub](https://github.com/MechaCritter/Python-Visual-Similarity/issues) with:
- A short description of the problem or feature request.
- Steps to reproduce (for bugs).
- Your Python version, OS, the **torch** version, and, if applicable, the **CUDA driver** version.

## Set up developer environment

This project uses [uv](https://github.com/astral-sh/uv) instead of `pip` for managing dependencies and virtual environments. For an installation guide, please check out [Astral's official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/MechaCritter/Python-Visual-Similarity.git
cd Python-Visual-Similarity

# 2. Create a virtual environment and install all dependencies
uv venv .venv
uv pip install -e .

# 3. Set up pre-commit hooks
uv pip install pre-commit
pre-commit install

# 4. Check out your feature/bugfix branch
git switch -c my-branch
```

## Code style

- Use **snake_case** for variables and functions, **PascalCase** for classes.
- Use `reST` docstrings and remember to annotate parameters and return values. An example:

```python
def add(a: int, b: int) -> int:
    """Add two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The sum of a and b.
    """
    return a + b
```

## Get in touch

- Open an issue on [GitHub](https://github.com/MechaCritter/Python-Visual-Similarity/issues).
- Email: [vunhathuy234@gmail.com](mailto:vunhathuy234@gmail.com)
- LinkedIn: [Nhat Huy Vu](https://www.linkedin.com/in/nhat-huy-vu-80495111b/)
