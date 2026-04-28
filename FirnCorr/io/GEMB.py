#!/usr/bin/env python
"""
GEMB.py
Written by Tyler Sutterley (04/2026)
Reads GEMB data products provided by Nichole Schlegel (NOAA)
    and Alex Gardner (JPL)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 04/2026
"""

from __future__ import print_function

import re
import gzip
import logging
import pathlib
import xarray as xr
import numpy as np
from FirnCorr.utilities import import_dependency, dependency_available

# attempt imports
dask = import_dependency("dask")
dask_available = dependency_available("dask")

# default variable attributes
_attributes = dict(
    zfirn=dict(
        standard_name="firn air content",
        long_name="Snow Height Change due to Compaction",
    ),
    zsmb=dict(long_name="Snow Height Change due to SMB"),
)

# PROJ4 parameters for GEMB model projections
# Antarctica: WGS84 / Polar Stereographic South
# Greenland: WGS84 / NSIDC Sea Ice Polar Stereographic North
proj4_params = dict(Antarctica=3031, Greenland=3413)


def open_mfdataset(
    filenames: list[str] | list[pathlib.Path],
    parallel: bool = False,
    **kwargs,
):
    """
    Open multiple files containing GEMB model data

    Parameters
    ----------
    filenames: list of str or pathlib.Path
        Path(s) to file(s) containing GEMB data
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    kwargs: dict
        Additional keyword arguments for opening GEMB files
    """
    # set default keyword arguments
    kwargs.setdefault("reference", None)
    kwargs.setdefault("range", None)
    # merge multiple granules
    if parallel and dask_available:
        opener = dask.delayed(open_dataset)
    else:
        opener = open_dataset
    # verify that filename is iterable
    if isinstance(filenames, str):
        filenames = [filenames]
    # read each file as xarray dataset and append to list
    d = [opener(f, **kwargs) for f in filenames]
    # read datasets as dask arrays
    if parallel and dask_available:
        (d,) = dask.compute(d)
    # merge variables from multiple files
    ds = xr.merge(d, compat="override", join="override")
    # return xarray dataset
    return ds


def open_dataset(
    filename: str | pathlib.Path,
    chunks: str | None = None,
    **kwargs,
):
    """
    Open a netCDF4 file containing GEMB data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to netCDF4 file containing GEMB data
    chunks: str or None, default None
        Chunk size for ``xarray`` dataset
    compressed: bool, default False
        If True, read gzipped netCDF4 file
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # regular expression pattern for extracting parameters
    (region,) = re.findall(
        r"(Antarctica|Greenland)", pathlib.Path(filename).stem
    )
    # read the netCDF4-format file
    logging.debug(filename)
    if kwargs["compressed"]:
        # read gzipped netCDF4 file
        f = gzip.open(filename, "rb")
        tmp = xr.open_dataset(f, mask_and_scale=True, chunks=chunks)
    else:
        tmp = xr.open_dataset(filename, mask_and_scale=True, chunks=chunks)
    # output dataset
    ds = xr.Dataset()
    # decode time variables from (nominal) decimal years to datetime64[D]
    ds["time"] = ((tmp["time"] - 1970.0) * 365.0).astype("datetime64[D]")
    # extract x and y coordinate arrays
    ds["y"] = tmp["y"].copy()
    ds["x"] = tmp["x"].copy()
    # check if fields are available to derive FAC or SMB
    FAC = "centered_FAC" in tmp.data_vars and "dFAC" in tmp.data_vars
    SMB = "centered_SMB" in tmp.data_vars and "accum_SMB" in tmp.data_vars
    # calculate derived fields (if available)
    if FAC:
        # firn air content change
        ds["zfirn"] = tmp["dFAC"] + tmp["centered_FAC"]
        ds["zfirn"].attrs.update(_attributes["zfirn"])
        ds["zfirn"].attrs["units"] = tmp["dFAC"].attrs.get("units", "")
        ds["zfirn"].attrs["group"] = ["dFAC", "centered_FAC"]
    if SMB:
        # total surface mass balance anomalies
        tmp["SMB"] = tmp["accum_SMB"] + tmp["centered_SMB"]
        # calculate original SMB data from anomalies
        SMB = tmp["SMB"].diff(dim="time", n=1, label="upper")
        SMB = SMB.reindex(time=tmp["SMB"].time, fill_value=0.0)
        SMB = SMB.assign_coords(time=ds["time"].values)
        # height change due to SMB
        ds["zsmb"] = SMB.copy()
        ds["zsmb"].attrs.update(_attributes["zsmb"])
        ds["zsmb"].attrs["units"] = tmp["accum_SMB"].attrs.get("units", "")
        ds["zsmb"].attrs["group"] = ["accum_SMB", "centered_SMB"]
    # verify mask is applied evenly for all time steps
    for var in ds.data_vars:
        ds[var] = ds[var].where(
            ds[var].notnull().all(dim="time"), np.nan, drop=False
        )
    # # verify that chunks are unified (if specified)
    if chunks is not None:
        ds = ds.unify_chunks()
    # add attributes to dataset
    ds.attrs["crs"] = proj4_params[region]
    # return the dataset
    return ds
