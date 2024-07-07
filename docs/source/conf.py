"""Configuration file for the Sphinx documentation builder."""

# for autodoc
from os import path as os_path
from sys import path as sys_path

sys_path.insert(0, os_path.abspath("../../"))

# -- Project information -----------------------------------------------------

# pylint: disable=invalid-name, redefined-builtin
project = "corr-im"
copyright = "2024, corr-im authors"
author = "corr-im authors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.imgmath",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "groundwork"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# preamble for latex
imgmath_image_format = "svg"  # default png
imgmath_font_size = 18  # default 12
imgmath_latex_preamble = """
\\usepackage{mathtools}
\\DeclarePairedDelimiterX{\\card}[1]\\lvert\\rvert{#1}
\\usepackage{xcolor}
\\color{white}
"""

# for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "igraph": ("https://python.igraph.org/en/stable/api/", None),
}
