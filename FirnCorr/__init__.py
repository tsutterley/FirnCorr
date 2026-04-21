"""
A firn correction library for Python
====================================

FirnCorr is a Python-based firn correction library for altimetry datasets.
It provides tools for reading and processing firn data using xarray.

The package works using scientific Python packages (numpy, scipy and pyproj)
combined with data storage in netCDF4 and HDF5 and mapping using
matplotlib and cartopy

Documentation is available at https://firncorr.readthedocs.io
"""

import FirnCorr.interpolate
import FirnCorr.spatial
import FirnCorr.utilities
import FirnCorr.version
from FirnCorr.regress import regress
from FirnCorr import io

# get version information
__version__ = FirnCorr.version.version
# read model database
models = io.load_database()
__models__ = sorted(models.keys())
