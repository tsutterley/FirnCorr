===============
Getting Started
===============

Firn Model Formats
##################

SMB and firn model files are available from different modeling groups in different formats.
``FirnCorr`` has drivers for GEMB, GSFC-fdm, MAR and RACMO model formats, which are presently usually netCDF4 files.
``FirnCorr`` uses ``pint`` to handle parsing the units of the model data and convert them into standard sets of units.

    - (:math:`m`): meters
    - (:math:`cm w.e.`): centimeters water equivalent 
    - (:math:`cm i.e.`): centimeters ice equivalent
    - (:math:`kg/m^2`): kilograms per square meter (equivalent to :math:`mm w.e.`)

Data Access
###########

Some model outputs can be programmatically downloaded using the fetching routines in ``FirnCorr.datasets``.

Other firn models may require manual downloading due to licensing agreements or limitations on programmatic access.
See the model links in :ref:`directories` for the references to specific firn models.

Model Database
##############

``FirnCorr`` stores the metadata for some models within a `JSON database <https://github.com/tsutterley/FirnCorr/blob/main/FirnCorr/data/database.json>`_.
``FirnCorr`` currently supports several solutions from the following models:

- ``GEMB`` :cite:t:`Gardner:2023gt`
- ``GSFC-fdm`` :cite:t:`Medley:2022ee`
- ``MAR``: :cite:t:`Fettweis:2017de,Tedesco:2020bi`
- ``RACMO-ascii`` :cite:t:`Ettema:2009ca`
- ``RACMO-netcdf`` :cite:t:`Noel:2018hk,vanWessem:2018jt`
- ``RACMO-downscaled`` :cite:t:`Noel:2019kn`

.. include:: Model-Database.ipynb
   :parser: myst_nb.docutils_

.. _directories:

Directories
###########

``FirnCorr`` uses a tree structure for storing and accessing the model outputs.
This structure was chosen based on the different formats of each model.
The base of the tree structure (in the table below as ``<model_path>``) can be the default ``FirnCorr`` cache directory or a user-specified (external) directory.
Several models can be programmatically downloaded from their providers to their parameterized directories using the fetching routines in ``FirnCorr.datasets``.

Presently, the following models and their directories are parameterized within ``FirnCorr``:

.. csv-table:: Antarctic Model Directories
   :file: ../_assets/ais-models.csv
   :header-rows: 1
   :width: 100%

.. csv-table:: Greenland Model Directories
   :file: ../_assets/gris-models.csv
   :header-rows: 1
   :width: 100%