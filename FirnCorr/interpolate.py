#!/usr/bin/env python
"""
interpolate.py
Written by Tyler Sutterley (05/2026)
Interpolators for spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 05/2026: added parameters to allow for extrapolation with
        inverse distance weighting (IDW) in addition to nearest-neighbors (NN)
    Updated 03/2026: break up extrapolation into separate functions to allow
        for caching of the kd-tree when interpolating multiple variables
    Updated 02/2026: output data from extrapolate as an xarray DataArray
        where there are no valid points within the cutoff distance
    Updated 08/2025: added a penalized least square inpainting function
    Written 12/2022
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import scipy.fftpack
import scipy.spatial
import FirnCorr.spatial


__all__ = [
    "inpaint",
    "extrapolate",
    "_to_cartesian",
    "_build_tree",
    "_query_tree",
]


def inpaint(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    N: int = 0,
    s0: int = 3,
    power: int = 2,
    epsilon: float = 2.0,
    **kwargs,
):
    """
    Inpaint over missing data in a two-dimensional array using a
    penalized least-squares method based on discrete cosine transforms
    :cite:p:`Garcia:2010hn,Wang:2012ei`

    Parameters
    ----------
    xs: np.ndarray
        x-coordinates
    ys: np.ndarray
        y-coordinates
    zs: np.ndarray
        Data with masked values
    N: int, default 0
        Number of iterations (0 for nearest neighbors)
    s0: int, default 3
        Smoothing factor
    power: int, default 2
        Power for lambda function
    epsilon: float, default 2.0
        Relaxation factor

    Returns
    -------
    z0: np.ndarray
        Data with inpainted (filled) values
    """
    # find masked values
    if isinstance(zs, np.ma.MaskedArray):
        W = np.logical_not(zs.mask)
    else:
        W = np.isfinite(zs)
    # no valid values can be found
    if not np.any(W):
        raise ValueError("No valid values found")

    # dimensions of input grid
    ny, nx = np.shape(zs)

    # calculate initial values using nearest neighbors
    # computation of distance Matrix
    # use scipy spatial KDTree routines
    xgrid, ygrid = np.meshgrid(xs, ys)
    tree = scipy.spatial.KDTree(np.c_[xgrid[W], ygrid[W]])
    # find nearest neighbors
    masked = np.logical_not(W)
    _, ii = tree.query(np.c_[xgrid[masked], ygrid[masked]], k=1)
    # copy valid original values
    z0 = np.zeros((ny, nx), dtype=zs.dtype)
    z0[W] = np.copy(zs[W])
    # copy nearest neighbors
    z0[masked] = zs[W][ii]
    # return nearest neighbors interpolation
    if N == 0:
        return z0

    # copy data to new array with 0 values for mask
    ZI = np.zeros((ny, nx), dtype=zs.dtype)
    ZI[W] = np.copy(z0[W])

    # calculate lambda function
    L = np.zeros((ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(ny) / ny)[:, None], (ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(nx) / nx)[None, :], (ny, nx))
    LAMBDA = np.power(2.0 * (2.0 - L), power)

    # smoothness parameters
    s = np.logspace(s0, -6, N)
    for i in range(N):
        # calculate discrete cosine transform
        GAMMA = 1.0 / (1.0 + s[i] * LAMBDA)
        DISCOS = GAMMA * scipy.fftpack.dctn(W * (ZI - z0) + z0, norm="ortho")
        # update interpolated grid
        z0 = (
            epsilon * scipy.fftpack.idctn(DISCOS, norm="ortho")
            + (1.0 - epsilon) * z0
        )

    # reset original values
    z0[W] = np.copy(zs[W])
    # return the inpainted grid
    return z0


# PURPOSE: extrapolate valid data to output points
def extrapolate(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    fill_value: float = None,
    cutoff: int | float = np.inf,
    is_geographic: bool = True,
    **kwargs,
):
    """
    Spatially extrapolate values beyond model domain using `KD-trees
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.spatial.KDTree.html>`_ and nearest-neighbor (NN)
    or inverse distance weighting (IDW)

    Parameters
    ----------
    xs: np.ndarray
        x-coordinates of firn model
    ys: np.ndarray
        y-coordinates of firn model
    zs: np.ndarray
        Firn model data
    X: np.ndarray
        Output x-coordinates
    Y: np.ndarray
        Output y-coordinates
    k: int, default 1
        Number of nearest neighbors to use for extrapolation
    fill_value: float, default np.nan
        Invalid value
    dtype: np.dtype, default np.float64
        Output data type
    cutoff: float, default np.inf
        Return only neighbors within distance (kilometers)

        Set to ``np.inf`` to extrapolate for all points
    is_geographic: bool, default True
        Input grid is in geographic coordinates

    Returns
    -------
    data: xr.DataArray
        Interpolated data
    """
    # calculate meshgrid of model coordinates
    gridx, gridy = np.meshgrid(xs, ys)
    # find valid values
    if isinstance(zs, np.ma.MaskedArray):
        indy, indx = np.nonzero(np.logical_not(zs.mask) & np.isfinite(zs.data))
    else:
        indy, indx = np.nonzero(np.isfinite(zs))
    # reduce to valid original values
    x0 = gridx[indy, indx]
    y0 = gridy[indy, indx]
    z0 = zs[indy, indx]
    # verify output dimensions
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    # extrapolate valid data values to data
    npts = len(X)
    # return none if no invalid points
    if npts == 0:
        return
    # calculate coordinates to query for neighboring points
    p_in = _to_cartesian(x0, y0, is_geographic=is_geographic)
    p_out = _to_cartesian(X, Y, is_geographic=is_geographic)
    # create KD-tree of valid points
    tree = _build_tree(p_in)
    # query output data points and extrapolate values
    data = _query_tree(
        tree, p_out, z0, k=k, cutoff=cutoff, fill_value=fill_value, **kwargs
    )
    # return the extrapolated data
    return data


def _to_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    is_geographic: bool = True,
):
    """
    Convert input coordinates to an array of points in a
    Cartesian coordinate system

    Parameters
    ----------
    x: np.ndarray
        x-coordinates to be converted
    y: np.ndarray
        y-coordinates to be converted
    is_geographic: bool, default True
        Coordinates are geographic

    Returns
    -------
    points: np.ndarray
        Output points in Cartesian coordinates
    """
    # verify output dimensions
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    # calculate coordinates to query for neighboring points
    if is_geographic:
        # global or regional equirectangular model
        # ellipsoidal major axis in kilometers
        a_axis = 6378.137
        # calculate Cartesian coordinates of input grid
        xi, yi, zi = FirnCorr.spatial.to_cartesian(x, y, a_axis=a_axis)
        # calculate Cartesian coordinates of output coordinates
        points = np.c_[xi, yi, zi]
    else:
        points = np.c_[x, y]
    # return the output points in Cartesian coordinates
    return points


def _build_tree(points: np.ndarray, **kwargs):
    """
    Build a KD-tree to search for neighboring points

    Parameters
    ----------
    points: np.ndarray
        Input points in Cartesian coordinates
    kwargs: dict
        Additional keyword arguments for ``scipy.spatial.KDTree``

    Returns
    -------
    tree: scipy.spatial.KDTree
        KD-tree from input points
    """
    # create KD-tree of points for nearest-neighbor extrapolation
    tree = scipy.spatial.KDTree(points, **kwargs)
    return tree


def _query_tree(
    tree: scipy.spatial.KDTree,
    points: np.ndarray,
    flattened: np.ndarray,
    k: int = 1,
    power: int = 2,
    cutoff: int | float = np.inf,
    fill_value: float = None,
    **kwargs,
):
    """
    Extrapolation of valid model data using KD-trees using
    nearest-neighbor (NN) or inverse distance weighting (IDW)

    Parameters
    ----------
    tree: scipy.spatial.KDTree
        KD-tree of valid points to query
    points: np.ndarray
        Output points in Cartesian coordinates
    flattened: np.ndarray
        Valid data array to be extrapolated
    k: int, default 1
        Number of nearest neighbors to use for extrapolation
    power: int, default 2
        Power for inverse distance weighting (IDW) extrapolation
    cutoff: float, default np.inf
        Return only neighbors within distance (kilometers)
    fill_value: float, default None
        Invalid value
    dtype: np.dtype, default from input data
        Output data type
    workers: int, default 1
        Number of parallel workers to use for KD-tree query

    Returns
    -------
    data: xr.DataArray
        Extrapolated data
    """
    # set default data type
    dtype = kwargs.get("dtype", flattened.dtype)
    workers = kwargs.get("workers", 1)
    # number of data points
    npts, _ = points.shape
    # query output data points and find k nearest neighbor within cutoff
    dd, ii = tree.query(
        points, k=k, distance_upper_bound=cutoff, workers=workers
    )
    # allocate to output extrapolate data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = data.fill_value
    # spatially extrapolate using nearest neighbors or IDW
    if k == 1 and np.any(np.isfinite(dd)):
        # spatially extrapolate using nearest neighbors
        (ind,) = np.nonzero(np.isfinite(dd))
        data.data[ind] = flattened[ii[ind]]
        data.mask[ind] = False
    elif k > 1 and np.any(np.isfinite(dd)):
        # clip distances to handle cases where points overlap
        # this can lead to infinite weights in the IDW extrapolation
        dd = np.clip(dd, a_min=1e-10, a_max=None)
        # clip indices to handle cases where there are fewer than k neighbors
        # weights will be nan so these points will be masked in the output
        ii = np.clip(ii, a_min=0, a_max=len(flattened) - 1)
        # normalized weights if power > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dd ** (-power)
        s = np.nansum(power_inverse_distance, axis=1)
        w = power_inverse_distance / np.broadcast_to(s[:, None], (npts, k))
        # spatially extrapolate using inverse distance weighting
        data.data[:] = np.nansum(w * flattened[ii], axis=1)
        data.mask[:] = np.logical_not(np.isfinite(dd).any(axis=1))
    # return extrapolated values
    return xr.DataArray(data)
