===
MAR
===

- Reads Modèle Atmosphérique Régional (MAR) data products provided by Lèige Université (Belgium)

Calling Sequence
----------------

.. code-block:: python

    import FirnCorr.io
    ds = FirnCorr.io.MAR.open_mfdataset(model_files, variable=["SMB", "ZN4", "ZN5", "ZN6"])

`Source code`__

.. __: https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/io/MAR.py


.. autofunction:: FirnCorr.io.MAR.open_mfdataset

.. autofunction:: FirnCorr.io.MAR.open_dataset
