# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from datetime import datetime, timezone
import sphinx_bootstrap_theme  # noqa: F401

# -- Project information -----------------------------------------------------

project = 'dicodile'
td = datetime.now(tz=timezone.utc)
copyright = (
    '2020-%(year)s, Dicodile Developers. Last updated %(short)s'
) % dict(year=td.year, iso=td.isoformat(),
         short=td.strftime('%Y-%m-%d %H:%M %Z'))

author = 'dicodile developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # extension to pull docstrings from modules to document
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # to generate automatic links to the documentation of
    # objects in other projects
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_pagenav': False,
    'source_link_position': "",
    'navbar_class': "navbar navbar-inverse",
    'navbar_links': [
        ("Examples", "auto_examples/index"),
        ("API", "api"),
        ("GitHub", "https://github.com/tomMoral/dicodile", True)
    ],
    'bootswatch_theme': "united"
}

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
