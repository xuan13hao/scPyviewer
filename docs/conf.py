"""Sphinx configuration for the scPyviewer documentation site.

Built with the pydata-sphinx-theme (the same theme scanpy / the scverse
ecosystem use), autosummary API reference generated from the package
docstrings, and MyST for Markdown + notebook pages.
"""
from __future__ import annotations

import importlib.metadata as _md

# -- Project information ------------------------------------------------------
project = "scPyviewer"
author = "Xuan Hao"
copyright = "2026, Xuan Hao"
try:
    release = _md.version("scPyviewer")
except _md.PackageNotFoundError:  # not installed (e.g. first local build)
    release = "0.6.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # parse the numpydoc-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",                   # Markdown (MyST) + Jupyter notebook pages
    "sphinx_design",             # grids / cards on the landing page
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbnails.db", ".DS_Store", "jupyter_execute"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

# numpydoc-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Don't re-execute notebooks during the docs build — ship their saved outputs.
nb_execution_mode = "off"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output (pydata-sphinx-theme, scverse-style) -------------------------
html_theme = "pydata_sphinx_theme"
html_title = "scPyviewer"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/xuan13hao/scPyviewer",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/scPyviewer/",
         "icon": "fa-brands fa-python"},
    ],
    "show_prev_next": False,
    "navigation_with_keys": False,
}
html_context = {
    "github_user": "xuan13hao",
    "github_repo": "scPyviewer",
    "github_version": "main",
    "doc_path": "docs",
}
