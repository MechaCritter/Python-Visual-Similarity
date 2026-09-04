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

## Cython modules

Some modules are implemented in Cython for performance. If you work on them,
please run this command to compile them:

```bash
make build-ext
```

## Working with vendored modules

Vendored files (files placed in folders named `_vendored`) are copied from their original sources. Checklist when vendoring
third-party repositories (See [hnswlib](pyvisim/image_store/_index/_vendored/README.md) for an example):

- [ ] The files remain unchanged from the original source and for
the rest of their lifetime inside `pyvisim`. If modifications are necessary, use class inheritance or overwrite methods in separate files.
- [ ] You have added a note in the `README.md` of the vendored folder and stated which files are copies of which file in the original source.
- [ ] You have created a copy of the license that the original source uses and placed it in `THIRD_PARTY/` folder.

## Downloaded test assets

Parts of the test suite need the Oxford Flowers dataset and the pretrained
torchvision backbones, which are downloaded on first use into the platform cache
directory (and into `TORCH_HOME` for the weights). You can fetch
them upfront to save some time in the CI:

```bash
uv run python .github/scripts/prefetch_assets.py
```

## Adding binary files to the docs

Binary files (images, diagrams, screenshots, benchmark barplots) are **not** committed to `main`. They must be committed to the orphan `assets` branch and referenced from the docs and the `README.md` through
absolute `https://raw.githubusercontent.com/.../assets/...` URLs. This prevents these files from bloating the repository history.

To add new binary files, place them under `docs/<topic>/` on that branch (e.g. `docs/architecture/`).

> [!WARNING]
> **Do not** add any file under `benchmarks/` as this
folder is wiped and regenerated on every benchmark run.

In short, you can run the following commands:

```bash
cd "$(mktemp -d)"
git clone -q --depth 1 --branch assets https://github.com/MechaCritter/Python-Visual-Similarity.git .

# add/replace whatever images you want here
mkdir -p docs/<topic> && cp ~/path/to/my-image.png docs/<topic>/

git checkout -q --orphan squashed
git add -A
git commit -q -m "Publish assets"
git push -f origin squashed:assets
```

Then link the image from the docs or the `README.md`:

```markdown
![My image](https://raw.githubusercontent.com/MechaCritter/Python-Visual-Similarity/assets/docs/<topic>/my-image.png)
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
