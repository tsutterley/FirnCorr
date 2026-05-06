======================
Setup and Installation
======================

Dependencies
############

``FirnCorr`` is dependent on several open source programs that can be installed using
OS-specific package management systems (e.g. ``apt`` or ``homebrew``),
``conda`` or from source:

- `PROJ <https://proj.org/>`_
- `HDF5 <https://www.hdfgroup.org/>`_
- `libxml2 <http://xmlsoft.org/>`_
- `libxslt <http://xmlsoft.org/XSLT/>`_

Installation
############

``FirnCorr`` is available for download from the `GitHub repository <https://github.com/tsutterley/FirnCorr>`_,
the `Python Package Index (pypi) <https://pypi.org/project/FirnCorr/>`_,
and from `conda-forge <https://anaconda.org/conda-forge/firncorr>`_.


The simplest installation for most users will likely be using ``conda`` or ``mamba``:

.. code-block:: bash

    conda install -c conda-forge firncorr

``conda`` installed versions of ``FirnCorr`` can be upgraded to the latest stable release:

.. code-block:: bash

    conda update firncorr

Development Install
###################

To use the development repository, please fork ``FirnCorr`` into your own account and then clone onto your system:

.. code-block:: bash

    git clone https://github.com/tsutterley/FirnCorr.git

``FirnCorr`` can then be installed within the package directory using ``pip``:

.. code-block:: bash

    python3 -m pip install --user .

To include all optional dependencies:

.. code-block:: bash

   python3 -m pip install --user .[all]

The development version of ``FirnCorr`` can also be installed directly from GitHub using ``pip``:

.. code-block:: bash

    python3 -m pip install --user git+https://github.com/tsutterley/FirnCorr.git

Package Management with ``pixi``
################################

Alternatively ``pixi`` can be used to create a `streamlined environment <https://pixi.sh/>`_ after cloning the repository:

.. code-block:: bash

    pixi install

``pixi`` maintains isolated environments for each project, allowing for different versions of
``FirnCorr`` and its dependencies to be used without conflict. The ``pixi.lock`` file within the
repository defines the required packages and versions for the environment.

``pixi`` can also create shells for running programs within the environment:

.. code-block:: bash

    pixi shell

To see the available tasks within the ``FirnCorr`` workspace:

.. code-block:: bash

    pixi task list

.. note::

    ``pixi`` is under active development and may change in future releases
