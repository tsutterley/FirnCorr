=======
GSFCfdm
=======

- Reads GSFC-fdm data products provided by Brooke Medley (NASA GSFC)

Calling Sequence
----------------

.. code-block:: python

    import FirnCorr.io
    ds = FirnCorr.io.GSFCfdm.open_dataset(model_file, variable=["FAC", "SMB_a", "h_a"])

`Source code`__

.. __: https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/io/GSFCfdm.py


.. autofunction:: FirnCorr.io.GSFCfdm.open_dataset

.. autofunction:: FirnCorr.io.GSFCfdm.decode_times
