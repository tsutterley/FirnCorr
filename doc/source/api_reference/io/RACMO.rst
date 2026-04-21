=====
RACMO
=====

- Reads Regional Atmospheric and Climate MOdel (RACMO) data products provided by IMAU (Utrecht University)

   * ``RACMO-ascii``
   * ``RACMO-downscaled``
   * ``RACMO-netcdf``

Calling Sequence
----------------

.. code-block:: python

    import FirnCorr.io
    ds = FirnCorr.io.RACMO.open_mfdataset(model_files, variable=["SMB"])

`Source code`__

.. __: https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/io/RACMO.py


.. autofunction:: FirnCorr.io.RACMO.open_mfdataset

.. autofunction:: FirnCorr.io.RACMO.open_dataset

.. autofunction:: FirnCorr.io.RACMO.open_ascii_dataset

.. autofunction:: FirnCorr.io.RACMO.open_netcdf_dataset

.. autofunction:: FirnCorr.io.RACMO.open_downscaled_dataset
