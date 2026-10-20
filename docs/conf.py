# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, os.path.abspath('..'))

project = 'pyAvalanches'
copyright = '2026, Morgane Marzulli'
author = 'Morgane Marzulli'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',          # Pull docstrings from code
    'sphinx.ext.napoleon',         # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',         # Add [source] links to docs
    'sphinx.ext.intersphinx',      # Link to external docs (numpy, etc.)
    'sphinx.ext.autosummary',      # Generate summary tables
    'sphinx_gallery.gen_gallery',  # FOr example gallery
]

# Napoleon settings (for Google/NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Intersphinx: link to other projects' docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'mne_bids': ("https://mne.tools/mne-bids/stable/", None),
}

# Autodoc settings
autodoc_member_order = 'bysource'      # Keep order as in source code
autodoc_typehints = 'description'      # Show type hints in description


templates_path = ['_templates']
exclude_patterns = []

# aggiunto per la nuova doc
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autoclass_content = "class"

#########

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'shibuya'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

sphinx_gallery_conf = {
    # Path to your example scripts (relative to conf.py)
    'examples_dirs': ['../examples'],

    # Where to put the generated output (relative to conf.py)
    'gallery_dirs': ['auto_examples'],

    # Which files to execute (only files starting with "plot_")
    'filename_pattern': r'/example_',

    # Thumbnail for scripts that don't generate a figure
    'default_thumb_file': '_static/default_thumb.png',

    # Order examples by filename
    'within_subsection_order': 'FileNameSortKey',

    # Download links for each example
    'download_all_examples': True,

    # matplotlib backend for rendering
    'matplotlib_animations': True,
}