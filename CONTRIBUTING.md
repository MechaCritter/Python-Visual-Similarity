# Contributing to pyvisim

Thank you for your interest! Contributions of all kinds are welcome.


## PR TODO list - your first PR

To understand how the library is structured as well the technical details before diving in, you can first read the [architecture documentation](docs/arc42.md), and/or you can also read the docstrings of the modules and classes that you are working on.

Use this checklist to stay on track for your first code PR:

- **Fork and clone this repository**: see [Set up developer environment](#set-up-developer-environment) section.
- **Check out the coding style**: see [Code style](#code-style) section.
- **Run tests**: run `make test-types` and `make fmt` before you make a PR. If `make test-types` stops on a syntax error inside NumPy, see [Known type-check issues](docs/ci/known_errors.md).
- **Add a release note**: run `make release-note NAME=my-change` and fill in the generated file, see [Release notes](#release-notes).
- **Open a PR** on GitHub.

## Using AI to contribute

I know, we all use Claude/Codex/OpenClaw and co. to help us write code faster. I am no exception. Just make sure that you review the generated code carefully before you make your PR.

> [!IMPORTANT]
> It is not difficult to detect an AI-generated PR that was not reviewed at all, and I will have to reject such PRs immediately because it shows you did not take time checking what the AI wrote 🙂.

Please keep pull requests focused - **only one feature or fix per PR**! That would
make review faster.

## Release notes

Every PR must include a release note under `releasenotes/notes`.

PRs whose changes are limited to
tests, comments, docstrings or the CI can be labelled `ignore-for-release-notes`
by a maintainer to bypass the check.

To create one:

```bash
make release-note NAME=your-change
```

This writes `releasenotes/notes/your-change-<unique-id>.yaml`. Then, **delete the sections that do not apply** and fill in the rest:

```yaml
---
features:
  - |
    Implemented batch size for ``ClipEmbedder``.
performance:
  - |
    Embedding all train images in the Oxford Flowers dataset went from
    60 s to 30 s on the GPU with ``PYVISIM_NUM_THREADS=4``.

    Specs: NVIDIA GeForce RTX 3090, CUDA 12.2, torch 2.7.1, Intel Core i9-13900K, 32 GB RAM, Ubuntu 22.04.3 LTS.
```

The sections are `highlights`, `upgrade`, `features`, `enhancements`,
`performance`, `issues`, `deprecations`, `security` and `fixes`. Use `upgrade`
for breaking changes, and say how a user can tell whether they are affected and
what to do about it.

Each section is rendered as
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). If
you are not yet familiar with this format, feel
free to check out the provided link.

> [!TIP]
> You can also write your release note in **Markdown** and convert it to **reStructuredText** with `pandoc` if you are more comfortable with Markdown:
>
> ```bash
> pandoc -f markdown -t rst -o releasenotes/notes/your-change-<unique-id>.rst releasenotes/notes/your-change-<unique-id>.md
> ```

### Writing code in `.rst` format

For inline code, use double backticks:

```
``ClipEmbedder``
```

For code blocks, use the
[code directive](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-code-block):

```
.. code:: python

    from pyvisim.neural_networks import ClipEmbedder

    embedder = ClipEmbedder(batch_size=32)
```

> [!IMPORTANT]
> Run `pre-commit run --all-files` if the CI complains about formatting.

> [!IMPORTANT]
> Commit the note on the same branch as your code, so it is reviewed together with the change it describes.

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
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/Python-Visual-Similarity.git
cd Python-Visual-Similarity
git remote add upstream https://github.com/MechaCritter/Python-Visual-Similarity.git

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
> **Do not** add any file under `benchmark/` as this
folder is wiped and regenerated on every benchmark run.

> [!IMPORTANT]
> 1. Changes to `assets` go through a pull request via your fork, just like the `main` branch.
>
> 2. Start your branch from the `assets` branch of this repository, not from `main`.
>
> 3. When you open the pull request, select `assets` as the base branch, since GitHub suggests `main` by default.

A list of sample commands to add a new image to the this repository:

```bash
git fetch upstream assets
git switch -c docs/<topic>-assets FETCH_HEAD
git add <files>
# Prevents pre-commit hook from blocking commit
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "Add <topic> images"
git push origin docs/<topic>-assets
```

Then link the image from the docs or the `README.md`:

```markdown
![My image](https://raw.githubusercontent.com/MechaCritter/Python-Visual-Similarity/assets/docs/<topic>/my-image.png)
```

The link only resolves once your `assets` pull request is merged, so merge it before the pull request that links the image.

After a merge, the `Squash assets` workflow collapses `assets` back into a single commit. This rewrites the branch, so every other open pull request against it has to be rebased before it can be merged. To move your branch onto the new `assets`, replace `<n>` with the number of commits on your branch (check with `git log --oneline`):

```bash
git fetch upstream assets
git rebase --onto FETCH_HEAD HEAD~<n>
git push -f origin docs/<topic>-assets
```

> [!IMPORTANT]
> Sometimes, GitHub caches old images, so your image might not be visible directly after pushing to `assets`. In this case, a new run of the `docs.yml` workflow is necessary.

## Documentation

Module documentation lives under `docs/<module>/` and is written in
**reStructuredText**, with the following structure:

```
docs/
  <module>/
    index.rst                  # the module page and its table of contents
    arc42.md                   # software architecture of the module
    <class>/
      <class>.rst              # one page per public class
      benchmark.md             # generated, only where there is a benchmark
```

Following types of documents are written in Markdown, on the other hand:

- **`arc42.md`** describes the software architecture of that
  module, including Architecture Decision Records.
- **`benchmark.md`** is generated by a script and is included into the class
  page it belongs to. **Do not edit it by hand** 😄

Build the documentation before you open a PR:

```bash
make docs
```

Then, open `docs/_build/html/index.html` to review the result.

### Tutorials

Tutorials are written as Jupyter Notebooks and will be added to the section
`tutorials/` in the documentation after every merge. See [the guidelines
here](docs/tutorials/notebooks/README.md).

> [!IMPORTANT]
>
> If you made API changes unrelated to the tutorials, some tutorials might become outdated and hence break. To save us precious CI time, please run `make test-notebooks` locally first before you commit and fix the affected tutorials.

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
