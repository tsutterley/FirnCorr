=======
spatial
=======

- Spatial transformation routines
- Gravitational and ellipsoidal parameters :cite:p:`HofmannWellenhof:2006hy,Petit:2010tp`

`Source code`__

.. __: https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/spatial.py

General Methods
===============

.. autofunction:: FirnCorr.spatial.data_type

.. autoclass:: FirnCorr.spatial.datum
   :members:

.. autofunction:: FirnCorr.spatial.convert_ellipsoid

.. autofunction:: FirnCorr.spatial.compute_delta_h

.. autofunction:: FirnCorr.spatial.wrap_longitudes

.. autofunction:: FirnCorr.spatial.to_cartesian

.. autofunction:: FirnCorr.spatial.to_sphere

.. autofunction:: FirnCorr.spatial.to_geodetic

.. autofunction:: FirnCorr.spatial._moritz_iterative

.. autofunction:: FirnCorr.spatial._bowring_iterative

.. autofunction:: FirnCorr.spatial._zhu_closed_form

.. autofunction:: FirnCorr.spatial.geocentric_latitude

.. autofunction:: FirnCorr.spatial.scale_factors
