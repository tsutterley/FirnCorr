#!/usr/bin/env python
"""
regress.py
Written by Tyler Sutterley (07/2022)
Estimates a time series for extrapolation by least-squares regression

CALLING SEQUENCE:
    d_out = regress(t_in, d_in, t_out, order=2,
        cycles=[0.25,0.5,1.0,2.0,4.0,5.0], relative=t_in[0])

INPUTS:
    t_in: input time array
    d_in: input data array
    t_out: output time array for calculating modeled values

OUTPUTS:
    d_out: reconstructed time series

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)

UPDATE HISTORY:
    Updated 07/2022: updated docstrings to numpy documentation format
    Updated 05/2021: define int/float precision to prevent deprecation warning
    Updated 07/2020: added function docstrings
    Written 07/2019
"""

import numpy as np


def regress(
    t_in: np.ndarray,
    d_in: np.ndarray,
    t_out: np.ndarray,
    order: int = 2,
    cycles: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 5.0],
    relative: float or list or Ellipsis = Ellipsis,
):
    """
    Fits a synthetic signal to data over a time period by
        ordinary or weighted least-squares

    Parameters
    ----------
    t_in: float
        Time array
    d_in: float
        Data array
    order: int, default 2
        Maximum polynomial order in fit

            * ``0``: constant
            * ``1``: linear
            * ``2``: quadratic
    cycles: list, default [0.25,0.5,1.0,2.0,4.0,5.0]
        Cyclical terms
    relative: float or List, default Ellipsis
        Epoch for calculating relative dates

            - ``float``: use exact value as epoch
            - ``list``: use mean from indices of available times
            - ``Ellipsis``: use mean of all available times

    Returns
    -------
    d_out: float
        Reconstructed time series
    """

    # remove singleton dimensions
    t_in = np.squeeze(t_in)
    d_in = np.squeeze(d_in)
    t_out = np.squeeze(t_out)
    # check dimensions of output
    t_out = np.atleast_1d(t_out)
    # calculate epoch for calculating relative times
    if isinstance(relative, (list, np.ndarray)):
        t_rel = t_in[relative].mean()
    elif isinstance(relative, (float, int, np.float_, np.int_)):
        t_rel = np.copy(relative)
    elif relative == Ellipsis:
        t_rel = t_in[relative].mean()

    # create design matrix based on polynomial order and harmonics
    DMAT = []
    MMAT = []
    # add polynomial orders (0=constant, 1=linear, 2=quadratic)
    for o in range(order + 1):
        DMAT.append((t_in - t_rel) ** o)
        MMAT.append((t_out - t_rel) ** o)
    # add cyclical terms (0.5=semi-annual, 1=annual)
    for c in cycles:
        DMAT.append(np.sin(2.0 * np.pi * t_in / np.float64(c)))
        DMAT.append(np.cos(2.0 * np.pi * t_in / np.float64(c)))
        MMAT.append(np.sin(2.0 * np.pi * t_out / np.float64(c)))
        MMAT.append(np.cos(2.0 * np.pi * t_out / np.float64(c)))

    # Calculating Least-Squares Coefficients
    # Standard Least-Squares fitting (the [0] denotes coefficients output)
    beta_mat = np.linalg.lstsq(np.transpose(DMAT), d_in, rcond=-1)[0]

    # return modeled time-series
    return np.dot(np.transpose(MMAT), beta_mat)
