"""Sphinx configuration for Irish Statute Assistant documentation."""
import os
import sys

# Make the src package importable for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "Irish Statute Assistant"
copyright = "2026"
author = ""
release = "1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
html_title = "Irish Statute Assistant"

myst_enable_extensions = ["colon_fence"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Suppress noisy MyST warnings about heading levels
suppress_warnings = ["myst.header"]
