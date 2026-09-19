# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# Local extensions live in docs/_ext/.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyvisim"
copyright = "2026, Nhật Huy Vũ"
author = "Nhật Huy Vũ"

release = importlib.metadata.version("pyvisim")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # Renders the tutorial notebooks. It sets up myst_parser itself, which parses
    # the Markdown pulled in via '.. include::'.
    "myst_nb",
    # Turns the alert blocks that GitHub renders in the notebooks into the
    # matching admonitions.
    "github_alerts",
    # Generates one page per release from releasenotes/notes/. reno reads the
    # notes of each version from the git history and its tags, so the
    # documentation has to be built from a git checkout.
    "reno_pages",
]

templates_path = ["_templates"]
# Markdown never stands on its own here: it is either pulled into an rST page
# via '.. include::' ('benchmark.md') or it is developer documentation that the
# published site does not carry ('arc42.md', 'testing/', 'benchmarks/').
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*.md",
    "**/*.md",
    # Autosave copies that Jupyter writes next to the tutorial notebooks.
    "**/.ipynb_checkpoints",
]

language = "en"

# The narrative pages included from docs/ link to repository source files
# (e.g. ../pyvisim/classic/vlad.py), which have no HTML equivalent.
# Their H1 titles are skipped via ':start-line: 1' so that each rST page owns
# the top-level heading, which makes the included content start at H2.
# A notebook rendered with 'nb_execution_mode=off' carries no kernel metadata to
# pick a lexer from, so its code cells fall back to the default highlighting.
suppress_warnings = ["myst.xref_missing", "myst.header", "myst-nb.lexer"]

# -- Autodoc -------------------------------------------------------------------

# The 'nn' extra is not required to build the documentation: any optional
# dependency that is missing from the environment is mocked so that autodoc can
# still import every module.
autodoc_mock_imports = [
    module
    for module in ("torch", "torchvision")
    if importlib.util.find_spec(module) is None
]

autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- MyST (Markdown) -----------------------------------------------------------

# The Sphinx sources are reStructuredText. MyST parses the Markdown cells of the
# tutorial notebooks and the narrative Markdown pages under docs/ that are
# pulled in via '.. include:: ... :parser: myst_parser.sphinx_'.
# dollarmath renders the $$...$$ formulas
myst_enable_extensions = ["colon_fence", "dollarmath"]
# Generate anchors for headings so links such as 'vlad.md#section' resolve.
myst_heading_anchors = 3

# -- MyST-NB (notebooks) -------------------------------------------------------

# The tutorial notebooks are committed without outputs. The build executes them
# and caches their outputs in '_build/.jupyter_cache', keyed on the code cells,
# so a notebook only runs again once its code changes.
nb_execution_mode = "cache"
# A notebook that raises fails the build instead of rendering the traceback.
nb_execution_raise_on_error = True
nb_execution_show_tb = True
# Training cells run for many minutes, so no cell timeout applies. A hung
# notebook is stopped by the timeout of the CI job instead.
nb_execution_timeout = -1

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

STATIC_DIR = Path(__file__).parent / "_static"
LIGHT_LOGO = "pyvisim-symbol-light.png"
DARK_LOGO = "pyvisim-symbol-dark.png"

GITHUB_URL = "https://github.com/MechaCritter/Python-Visual-Similarity"

# The GitHub mark, from the Octicons set (MIT licensed), rendered as the
# clickable icon in the page footer.
GITHUB_ICON = """
<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
    1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
    .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
    0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38
    A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
</svg>
"""

html_theme = "furo"
html_title = f"pyvisim {release}"
html_static_path = ["_static"]
# The tutorial chapters and their notebooks are numbered, and the numbers read
# as '1.1 Title' rather than '1.1. Title'.
html_secnumber_suffix = " "

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": GITHUB_URL,
            "html": GITHUB_ICON,
            "class": "",
        },
    ],
}

# The logo files are binaries, so they are published under 'docs/logo' on the
# orphan 'assets' branch and copied into '_static' by the 'Docs' workflow.
# A build without that copy, such as a fresh clone, keeps the plain title
# instead of pointing furo at a file that is not there.
if (STATIC_DIR / LIGHT_LOGO).is_file() and (STATIC_DIR / DARK_LOGO).is_file():
    html_theme_options["light_logo"] = LIGHT_LOGO
    html_theme_options["dark_logo"] = DARK_LOGO
